"""Independent audio encoder training; selection is exclusively on validation macro-F1."""
import argparse
import csv
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import yaml

from .data import LABELS, manifest, extract, collate
from .models import AudioEncoder

KINDS = ('cnn', 'bilstm', 'transformer')

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred):
    """Fixed eight-class macro averages, including absent classes with zero division=0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0 or y_true.shape != y_pred.shape:
        raise ValueError('Nonempty matched targets and predictions required')
    if not np.isin(y_true, range(8)).all() or not np.isin(y_pred, range(8)).all():
        raise ValueError('Labels must be in 0..7')
    cm = np.zeros((8,8), dtype=np.int64)
    np.add.at(cm, (y_true.astype(int),y_pred.astype(int)), 1)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(8), where=cm.sum(0)>0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(8), where=cm.sum(1)>0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(8), where=(precision+recall)>0)
    return dict(accuracy=float(tp.sum()/cm.sum()), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                confusion_matrix=cm.tolist(),
                per_class={name:dict(precision=float(precision[i]),recall=float(recall[i]),
                                    f1=float(f1[i]),support=int(cm[i].sum())) for i,name in enumerate(LABELS)})


def fit_standardizer(features):
    # Frames from clean TRAIN samples only, float64 accumulation for stability.
    total = sum(len(x) for x in features)
    if total < 1:
        raise ValueError('No training frames')
    sums = sum((x.double().sum(0) for x in features))
    squares = sum((x.double().square().sum(0) for x in features))
    mean = sums / total
    std = (squares/total - mean.square()).clamp_min(1e-10).sqrt()
    return mean.float(), std.float()


class TrainingDataset(Dataset):
    """Identical order/augmentation per sample and epoch across independent encoders."""
    def __init__(self, rows, preprocessing, clean, split, augment, seed):
        if split not in ('train','val') or (augment and split != 'train'):
            raise ValueError('Training stage accepts only train/val; augmentation only train')
        self.rows = [r for r in rows if r['split'] == split]
        self.preprocessing, self.clean = preprocessing, clean
        self.augment, self.seed, self.epoch = augment, seed, 0
        self.augment_probability=1.0
        self.cache=None
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        gate=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index,1]))
        if self.augment and (self.augment_probability>=1 or gate.random()<self.augment_probability):
            if self.cache is not None:
                features=self.cache.get(self.epoch,index)
            else:
                rng = np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,index]))
                features = extract(row['path'],self.preprocessing,'train',augment=True,rng=rng)
        else:
            features = self.clean[row['sample_id']]
        return dict(row, features=features)


def run_epoch(model, loader, device, optimizer=None, grad_clip=1.):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0., 0
    truth, predictions, records = [], [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            expected = 'train' if training else 'val'
            if any(s != expected for s in batch['splits']):
                raise ValueError(f'{expected} loader contains another split')
            x, lengths = batch['x'].to(device), batch['lengths'].to(device)
            target = batch['labels'].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x,lengths)['logits']
            loss = nn.functional.cross_entropy(logits,target,**(getattr(model,'loss_options',{}) if training else {}))
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            probs = logits.detach().softmax(-1).cpu()
            pred = probs.argmax(-1).tolist()
            labels = target.cpu().tolist()
            truth.extend(labels); predictions.extend(pred)
            loss_sum += loss.item()*len(labels); count += len(labels)
            if not training:
                records.extend(dict(sample_id=s,split='val',label=y,prediction=hat,
                                    **{f'p_{name}':float(prob[i]) for i,name in enumerate(LABELS)})
                               for s,y,hat,prob in zip(batch['sample_ids'],labels,pred,probs))
    result = metrics(truth,predictions)
    result['loss'] = loss_sum / count
    return result, records


def save_checkpoint(path, payload):
    # Atomic replacement of our own files; each run directory is created exclusively.
    temporary = Path(str(path)+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def train_one(kind, train_ds, val_ds, config, preprocessing, mean, std, run_dir, provenance):
    seed_all(config['seed'])
    model = AudioEncoder(kind, input_dim=preprocessing['n_mfcc'])
    initialization=None
    if config.get('warm_start'):
        parent=Path(config['warm_start'])
        if json.loads((parent/'status.json').read_text()).get('status')!='completed': raise ValueError('Warm-start run is incomplete')
        previous=parent/kind/'best.pt'
        saved=torch.load(previous,map_location='cpu',weights_only=True)
        if saved['encoder']!=kind or saved['class_names']!=LABELS or saved['preprocessing']!=preprocessing or saved['provenance'].get('diagnostic'):
            raise ValueError('Incompatible warm-start checkpoint')
        if saved['provenance']['split_sha256']!=provenance['split_sha256']: raise ValueError('Warm-start actor split mismatch')
        model.load_state_dict(saved['model_state_dict'],strict=True)
        initialization={'checkpoint':str(previous),'sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),'parent_epoch':saved['epoch']}
    model.feature_mean.copy_(mean); model.feature_std.copy_(std)
    device = torch.device(config['device'])
    model.to(device)
    model.head[2].p=config.get('head_dropout',0.2)
    model.loss_options={'label_smoothing':config.get('label_smoothing',0.0)}
    if config.get('class_balance',False):
        counts=torch.bincount(torch.tensor([r['label'] for r in train_ds.rows]),minlength=8).float()
        if (counts==0).any(): raise ValueError('Class balancing requires all eight training classes')
        model.loss_options['weight']=(counts.sum()/(8*counts)).to(device)
    train_ds.augment_probability=config.get('augment_probability',1.0)
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=config['scheduler_factor'],patience=config['scheduler_patience'])
    generator = torch.Generator().manual_seed(config['seed'])
    train_loader = DataLoader(train_ds,batch_size=config['batch_size'],shuffle=True,collate_fn=collate,
                              num_workers=config['num_workers'],generator=generator)
    val_loader = DataLoader(val_ds,batch_size=config['batch_size'],shuffle=False,collate_fn=collate,
                            num_workers=config['num_workers'])
    destination = run_dir/kind
    destination.mkdir()
    best, best_epoch, stale, history = -1., 0, 0, []
    start = time.monotonic()
    for epoch in range(1,config['epochs']+1):
        train_ds.epoch = epoch
        training, _ = run_epoch(model,train_loader,device,optimizer,config['grad_clip'])
        validation, records = run_epoch(model,val_loader,device)
        current_lr = optimizer.param_groups[0]['lr']
        improved = validation['macro_f1'] > best
        if improved:
            best, best_epoch, stale = validation['macro_f1'], epoch, 0
        else:
            stale += 1
        scheduler.step(validation['macro_f1'])
        history.append(dict(epoch=epoch,learning_rate=current_lr,
                            train={k:training[k] for k in ('loss','accuracy','macro_f1')},
                            val={k:validation[k] for k in ('loss','accuracy','macro_precision','macro_recall','macro_f1')}))
        payload = dict(format_version=2,initialization=initialization,encoder=kind,epoch=epoch,best_epoch=best_epoch,best_val_macro_f1=best,
                       model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                       scheduler_state_dict=scheduler.state_dict(),torch_rng_state=torch.get_rng_state(),
                       train_generator_state=generator.get_state(),train_config=config,preprocessing=preprocessing,
                       class_names=LABELS,provenance=provenance,validation_metrics=validation)
        save_checkpoint(destination/'last.pt',payload)
        if improved:
            save_checkpoint(destination/'best.pt',payload)
            write_json(destination/'best_val_metrics.json',dict(epoch=epoch,**validation))
            with (destination/'best_val_confusion_matrix.csv').open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(['true/predicted',*LABELS])
                writer.writerows([name,*row] for name,row in zip(LABELS,validation['confusion_matrix']))
            with (destination/'best_val_predictions.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        write_json(destination/'history.json',history)
        with (destination/'history.csv').open('w',newline='') as f:
            flat=[dict(epoch=h['epoch'],learning_rate=h['learning_rate'],
                       **{f'train_{k}':v for k,v in h['train'].items()},
                       **{f'val_{k}':v for k,v in h['val'].items()}) for h in history]
            writer=csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        print(f'{kind:11} epoch={epoch:02d} train_loss={training["loss"]:.4f} val_loss={validation["loss"]:.4f} val_macro_f1={validation["macro_f1"]:.4f} best={best:.4f}',flush=True)
        if stale >= config['patience']:
            break
    summary = dict(encoder=kind,initialization=initialization,best_epoch=best_epoch,best_val_macro_f1=best,epochs_completed=len(history),
                   elapsed_seconds=round(time.monotonic()-start,2),checkpoint=str((destination/'best.pt').relative_to(run_dir)),
                   protocol_sha256=provenance['protocol_sha256'],diagnostic=provenance['diagnostic'],
                   validation_metrics=json.loads((destination/'best_val_metrics.json').read_text()))
    write_json(destination/'summary.json',summary)
    return summary


def compare(results, destination, diagnostic=False):
    if len(results) != 3 or {r['encoder'] for r in results} != set(KINDS):
        raise ValueError('Comparison requires all three independent encoders')
    if len({r['protocol_sha256'] for r in results}) != 1:
        raise ValueError('Different comparison protocols')
    ranking = sorted(results,key=lambda r:(-r['best_val_macro_f1'],KINDS.index(r['encoder'])))
    tied = [r['encoder'] for r in ranking if r['best_val_macro_f1']==ranking[0]['best_val_macro_f1']]
    table = [dict(encoder=r['encoder'],best_epoch=r['best_epoch'],
                  **{k:r['validation_metrics'][k] for k in ('macro_f1','accuracy','macro_precision','macro_recall')},
                  checkpoint=r['checkpoint']) for r in ranking]
    report = dict(selection_metric='validation_macro_f1',test_evaluated=False,diagnostic=diagnostic,
                  selected_encoder=None if diagnostic else ranking[0]['encoder'],
                  selected_checkpoint=None if diagnostic else ranking[0]['checkpoint'],
                  tied_encoders=tied,tie_rule='fixed candidate order cnn, bilstm, transformer',comparison=table)
    write_json(destination/'comparison.json',report)
    with (destination/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return report


def validate_config(config):
    for k in ('epochs','batch_size','patience','scheduler_patience','threads'):
        if not isinstance(config[k],int) or config[k]<1:
            raise ValueError(f'{k} must be a positive integer')
    if not isinstance(config['augment'],bool):
        raise ValueError('augment must be a YAML boolean')
    if not isinstance(config['seed'],int) or config['seed']<0:
        raise ValueError('seed must be a nonnegative integer')
    if config['num_workers'] != 0:
        raise ValueError('Use num_workers=0 for reproducible epoch-specific online augmentation')
    if config['learning_rate']<=0 or config['weight_decay']<0 or config['grad_clip']<=0 or not 0<config['scheduler_factor']<1:
        raise ValueError('Invalid optimization configuration')
    for name,default in [('augment_probability',1.0),('label_smoothing',0.0),('head_dropout',0.2)]:
        if not 0<=config.get(name,default)<=(1.0 if name=='augment_probability' else .9):
            raise ValueError(f'Invalid {name}')
    if not isinstance(config.get('class_balance',False),bool): raise ValueError('class_balance must be boolean')
    if config['device'] not in ('cpu','mps','cuda'):
        raise ValueError('device must be cpu, mps or cuda')



def candidate_worker(kind,eligible,clean_numpy,config,preprocessing,mean,std,root,provenance,cache_root):
    # Spawned processes use independent RNG state; numerical protocol is unchanged.
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    clean={k:torch.from_numpy(v) for k,v in clean_numpy.items()}
    train=TrainingDataset(eligible,preprocessing,clean,'train',config['augment'],config['seed'])
    val=TrainingDataset(eligible,preprocessing,clean,'val',False,config['seed'])
    if cache_root and config['augment']:
        from .augmentation_cache import EpochAugmentationCache
        train.cache=EpochAugmentationCache(train.rows,preprocessing,config['seed'],cache_root)
    return train_one(kind,train,val,config,preprocessing,torch.from_numpy(mean),torch.from_numpy(std),Path(root),provenance)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder',choices=(*KINDS,'all'),default='all')
    parser.add_argument('--data',default='data/raw')
    parser.add_argument('--splits',default='metadata/actor_splits.csv')
    parser.add_argument('--preprocessing',default='configs/default.yaml')
    parser.add_argument('--config',default='configs/train.yaml')
    parser.add_argument('--output',required=True,help='New output directory; existing directories are rejected')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--device',choices=('cpu','mps','cuda'))
    parser.add_argument('--warm-start',help='Initialize each encoder from the corresponding best checkpoint of a completed run; reset optimizer')
    parser.add_argument('--cache',help='Optional train-only augmented MFCC cache; does not change numerical inputs')
    parser.add_argument('--smoke',action='store_true',help='Two epochs, 16 train + 16 val records; no best encoder selection')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    preprocessing=yaml.safe_load(Path(args.preprocessing).read_text())
    if args.epochs is not None: config['epochs']=args.epochs
    if args.device: config['device']=args.device
    if args.warm_start: config['warm_start']=args.warm_start
    if args.smoke:
        config.update(epochs=2,batch_size=8)
    validate_config(config)
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    seed_all(config['seed'])
    rows=manifest(args.data,args.splits)
    # Only inspect filenames for full integrity; never decode test audio or compute test predictions.
    eligible=[r for r in rows if r['split'] in ('train','val')]
    if args.smoke:
        # Balanced diagnostic subset, at most 2 samples per class per split.
        eligible=[r for split in ('train','val') for label in range(8)
                  for r in [r for r in eligible if r['split']==split and r['label']==label][:2]]
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=False)
    try:
        import shutil
        snapshot=destination/'source'; snapshot.mkdir()
        for name in ('train.py','models.py','data.py','augmentation_cache.py'):
            path=Path(__file__).with_name(name)
            if path.exists(): shutil.copy2(path,snapshot/name)
        print('Preparing clean train/validation MFCCs; test audio is excluded.',flush=True)
        clean={r['sample_id']:extract(r['path'],preprocessing,r['split']) for r in eligible}
        mean,std=fit_standardizer([clean[r['sample_id']] for r in eligible if r['split']=='train'])
        protocol=dict(training=config,preprocessing=preprocessing,split_sha256=hashlib.sha256(Path(args.splits).read_bytes()).hexdigest(),
                      source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('train.py','models.py','data.py')},
                      samples=[{k:r[k] for k in ('sample_id','split','label')} for r in eligible],class_names=LABELS,diagnostic=args.smoke)
        provenance=dict(protocol_sha256=fingerprint(protocol),diagnostic=args.smoke,split_sha256=protocol['split_sha256'])
        write_json(destination/'protocol.json',{**protocol,**provenance,'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__})
        write_json(destination/'normalization.json',dict(source='clean training frames only',mean=mean.tolist(),std=std.tolist()))
        train=TrainingDataset(eligible,preprocessing,clean,'train',config['augment'],config['seed'])
        val=TrainingDataset(eligible,preprocessing,clean,'val',False,config['seed'])
        if args.cache and config['augment']:
            from .augmentation_cache import EpochAugmentationCache
            train.cache=EpochAugmentationCache(train.rows,preprocessing,config['seed'],args.cache)
        results=[]
        if config.get('parallel_models',False) and args.encoder=='all':
            from concurrent.futures import ProcessPoolExecutor
            import multiprocessing
            clean_numpy={k:v.numpy() for k,v in clean.items()}
            with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(candidate_worker,kind,eligible,clean_numpy,config,preprocessing,mean.numpy(),std.numpy(),str(destination),provenance,args.cache) for kind in KINDS]
                results=[future.result() for future in futures]
        else:
            for kind in KINDS if args.encoder=='all' else (args.encoder,):
                results.append(train_one(kind,train,val,config,preprocessing,mean,std,destination,provenance))
        if args.encoder=='all':
            report=compare(results,destination,args.smoke)
            print(json.dumps(report,indent=2),flush=True)
        write_json(destination/'status.json',dict(status='completed',diagnostic=args.smoke,test_evaluated=False))
    except BaseException as error:
        write_json(destination/'status.json',dict(status='failed',error=repr(error),test_evaluated=False))
        raise

if __name__=='__main__': main()
