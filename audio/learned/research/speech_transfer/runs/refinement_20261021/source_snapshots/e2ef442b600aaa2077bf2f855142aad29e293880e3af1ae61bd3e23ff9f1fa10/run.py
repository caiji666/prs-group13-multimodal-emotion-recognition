"""One formal frozen-speech experiment, selected on validation Macro-F1 only."""
import argparse,csv,json,hashlib,random,time,copy,platform
from pathlib import Path
import numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from .models import build
from ..prepare import ROOT,LAYERS,REVISION
from ravdess_m2.data import LABELS
from ravdess_m2.train import metrics

def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_predictions(path,rows,probs):
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['sample_id','actor_id','split','label','prediction',*['p_'+x for x in LABELS]])
        w.writerows([r['sample_id'],r['actor_id'],'val',r['label'],int(p.argmax()),*p.tolist()] for r,p in zip(rows,probs))

class Inputs(Dataset):
    def __init__(self,rows,values):self.rows,self.values=rows,values
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.values[i],self.rows[i]['label']
def collate(items):
    x,y=zip(*items)
    if x[0].shape==torch.Size([3,1536]):return torch.stack(x),None,torch.tensor(y)
    lengths=torch.tensor([len(v) for v in x]);return torch.nn.utils.rnn.pad_sequence(x,batch_first=True),lengths,torch.tensor(y)

@torch.inference_mode()
def predict(model,loader):
    model.eval();out=[]
    for x,lengths,y in loader:out.append(model(x,lengths).softmax(-1).numpy())
    return np.concatenate(out)

def run(config,output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);dump(out/'config.json',config)
    torch.set_num_threads(4);seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    index=torch.load(ROOT/'cache/pooled_index.pt',map_location='cpu',weights_only=True)
    assert index['model_revision']==REVISION and set(r['split'] for r in index['rows'])=={'train','val'}
    assert index['split_sha256']==sha(ROOT.parents[3]/'multimodal_fusion/configs/actor_splits.csv'), 'Cached actor split changed'
    assert index['class_names']==LABELS and index['layers']==LAYERS
    assert index['model_sha256']==sha(ROOT/'assets/wavlm-base-plus/pytorch_model.bin'), 'Cached upstream model changed'
    rows=index['rows'];train_ix=[i for i,r in enumerate(rows) if r['split']=='train'];val_ix=[i for i,r in enumerate(rows) if r['split']=='val']
    assert len(train_ix)==960 and len(val_ix)==240
    train_rows=[rows[i] for i in train_ix];val_rows=[rows[i] for i in val_ix]
    if config['pooling']=='attentive_statistics':
        layer=LAYERS.index(config['layer']);values=[torch.load(ROOT/'cache'/(r['sample_id']+'.pt'),map_location='cpu',weights_only=True)['hidden'][layer].float() for r in rows]
        training_frames=torch.cat([values[i] for i in train_ix]);mean=training_frames.mean(0);std=training_frames.std(0,unbiased=False).clamp_min(1e-4)
        del training_frames
    else:
        values=list(index['pooled']);t=index['pooled'][train_ix];mean=t.mean(0);std=t.std(0,unbiased=False).clamp_min(1e-4)
    train=DataLoader(Inputs(train_rows,[values[i] for i in train_ix]),batch_size=config['batch_size'],shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(seed),num_workers=0)
    val=DataLoader(Inputs(val_rows,[values[i] for i in val_ix]),batch_size=32,shuffle=False,collate_fn=collate,num_workers=0)
    model=build(config,mean,std)
    if config.get('warm_start'):
        ck=torch.load(config['warm_start'],map_location='cpu',weights_only=True);
        loaded=model.load_state_dict(ck['state_dict'],strict=False)
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        assert all(any(tag in k for tag in ('frame_blocks','context','head.refinement')) for k in loaded.missing_keys), loaded.missing_keys
    freeze=config.get('freeze',[])
    for name,param in model.named_parameters():
        if any(name.startswith(prefix+'.') for prefix in freeze):param.requires_grad_(False)
    trainable=[p for p in model.parameters() if p.requires_grad]
    assert trainable
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=4)
    counts=torch.bincount(torch.tensor([r['label'] for r in train_rows]),minlength=8).float()
    weights=(counts.sum()/(8*counts)).pow(config.get('class_balance_power',.5)) if config.get('class_balance',True) else None
    best=-1;stale=0;history=[];start=time.monotonic();truth=[r['label'] for r in val_rows]
    for epoch in range(1,config['epochs']+1):
        model.train();total=0.
        for x,lengths,y in train:
            optimizer.zero_grad(set_to_none=True)
            features=model.encode(x,lengths)
            if config.get('mixup_alpha',0)>0:
                lam=float(np.random.beta(config['mixup_alpha'],config['mixup_alpha']));perm=torch.randperm(len(y))
                logits=model.head(lam*features+(1-lam)*features[perm])
                loss=lam*torch.nn.functional.cross_entropy(logits,y,weight=weights,label_smoothing=config.get('label_smoothing',0.))+(1-lam)*torch.nn.functional.cross_entropy(logits,y[perm],weight=weights,label_smoothing=config.get('label_smoothing',0.))
            else:loss=torch.nn.functional.cross_entropy(model.head(features),y,weight=weights,label_smoothing=config.get('label_smoothing',0.))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if config.get('focal_gamma',0):
                assert config.get('mixup_alpha',0)==0
                focal_logits=model.head(features)
                raw=torch.nn.functional.cross_entropy(focal_logits,y,weight=weights,reduction='none')
                with torch.no_grad():pt=focal_logits.softmax(-1).gather(1,y[:,None]).squeeze(1)
                loss=((1-pt).pow(config['focal_gamma'])*raw).mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=loss.item()*len(y)
        probs=predict(model,val);m=metrics(truth,probs.argmax(1));scheduler.step(m['macro_f1'])
        history.append(dict(epoch=epoch,train_loss=total/960,learning_rate=optimizer.param_groups[0]['lr'],**{f'val_{k}':m[k] for k in ('accuracy','macro_f1','macro_precision','macro_recall')}))
        if m['macro_f1']>best:
            best=m['macro_f1'];stale=0
            ck=dict(state_dict=model.state_dict(),config=config,epoch=epoch,validation_metrics=m,class_names=LABELS,upstream={'model_id':'microsoft/wavlm-base-plus','revision':REVISION,'sha256':index['model_sha256']},split_sha256=index['split_sha256'],normalization_fit_split='train',test_evaluated=False,upstream_frozen=True)
            torch.save(ck,out/'best.pt');dump(out/'best_val_metrics.json',dict(epoch=epoch,**m));write_predictions(out/'best_val_predictions.csv',val_rows,probs)
            with (out/'best_val_confusion_matrix.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['true/predicted',*LABELS]);w.writerows([name,*cm] for name,cm in zip(LABELS,m['confusion_matrix']))
        else:stale+=1
        dump(out/'history.json',history)
        with (out/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(history[0]),lineterminator='\n');w.writeheader();w.writerows(history)
        print(f"epoch={epoch:03d} loss={total/960:.4f} val_acc={m['accuracy']:.4f} val_F1={m['macro_f1']:.4f} best={best:.4f}",flush=True)
        if stale>=config['patience']:break
    best_ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best_ck['state_dict']);roundtrip=predict(model,val)
    re=metrics(truth,roundtrip.argmax(1));assert re['macro_f1']==best_ck['validation_metrics']['macro_f1']
    result=dict(config=config,completed_epochs=len(history),elapsed_seconds=time.monotonic()-start,best_epoch=best_ck['epoch'],**best_ck['validation_metrics'],checkpoint=str(out/'best.pt'),checkpoint_sha256=sha(out/'best.pt'),test_evaluated=False,upstream_frozen=True,train_samples=960,val_samples=240,checkpoint_roundtrip_verified=True)
    dump(out/'summary.json',result);dump(out/'status.json',dict(status='completed',test_evaluated=False,diagnostic=False))
    dump(out/'provenance.json',dict(refinement_source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},upstream=best_ck['upstream'],cache_metadata=json.loads((ROOT/'cache/metadata.json').read_text()),regime='frozen_encoder_then_classifier',mask_convention='True=padding',seed=seed,selection='validation Macro-F1'))
    print('RESULT',json.dumps({k:result[k] for k in ('accuracy','macro_f1','best_epoch','completed_epochs','elapsed_seconds')}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.output)
if __name__=='__main__':main()
