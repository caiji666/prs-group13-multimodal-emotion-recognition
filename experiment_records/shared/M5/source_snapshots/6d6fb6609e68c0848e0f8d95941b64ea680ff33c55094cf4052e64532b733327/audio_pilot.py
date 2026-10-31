"""Import M2 learned-only exports and run a bounded, validation-only audio pilot."""
import argparse
import csv
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from . import CLASSES
from .data import load_modality, trial_id
from .experiment import metrics, seed_all, write_json
from .models import head


def canonical_classes(names):
    names = tuple('surprise' if str(x) == 'surprised' else str(x) for x in names)
    if names != CLASSES:
        raise ValueError('Unexpected class order; refusing implicit probability reordering')
    return np.asarray(CLASSES)


def import_archive(archive, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source_root = destination / 'source'
    source_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Read data only; never import or execute content from the supplied archive.
        for info in z.infolist():
            target = (source_root / info.filename).resolve()
            if not target.is_relative_to(source_root.resolve()):
                raise ValueError('Unsafe archive path')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlink archive member rejected')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(info))
    source = source_root / 'm5_learned_feature_only'
    metadata = json.loads((source/'metadata.json').read_text())
    canonical_classes(metadata['class_names'])
    with (source/'manifest.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1440 or len({r['sample_id'] for r in rows}) != 1440:
        raise ValueError('Expected 1440 unique trials')
    actors, sequence_summary, converted = {}, {}, {}
    seen_actors = set()
    for split, expected_n in [('train',960),('val',240),('test',240)]:
        subset = [r for r in rows if r['split'] == split]
        if len(subset) != expected_n:
            raise ValueError(f'Unexpected count for {split}')
        actors[split] = sorted({int(r['actor_id']) for r in subset})
        if seen_actors.intersection(actors[split]):
            raise ValueError('Source actor leakage')
        seen_actors.update(actors[split])
        by_id = {r['sample_id']:r for r in subset}
        with np.load(source/'pooled'/f'{split}.npz',allow_pickle=False) as z:
            classes = canonical_classes(z['class_names'])
            ids = z['sample_ids'].copy()
            if set(ids.tolist()) != set(by_id) or len(ids) != len(by_id):
                raise ValueError('Manifest/pooled IDs mismatch')
            if not np.all(z['splits'] == split):
                raise ValueError('Pooled split mismatch')
            d = dict(sample_ids=ids,labels=z['labels'].copy(),actors=z['actor_ids'].copy(),
                     class_names=classes, pooled=z['F_audio_learned_feature_only'].copy(),
                     probs=z['P_audio_learned_feature_only'].copy(),data_kind=np.array('real_m2_learned_only'))
        if d['pooled'].shape != (expected_n,128):
            raise ValueError('Unexpected audio feature dimensions')
        lengths = []
        for i,sid in enumerate(ids):
            row = by_id[str(sid)]
            if trial_id('03-'+str(sid)+'.wav') != sid:
                raise ValueError('Invalid trial ID')
            expected_path = f'per_sample/{split}/{sid}.npz'
            if row['feature_file'] != expected_path:
                raise ValueError('Unexpected sample path')
            with np.load(source/expected_path,allow_pickle=False) as sample:
                canonical_classes(sample['class_names'])
                if (str(sample['sample_id']) != sid or str(sample['split']) != split
                    or int(sample['actor_id']) != int(row['actor_id'])):
                    raise ValueError('Sample metadata mismatch')
                if not (int(sample['actor_id']) == int(d['actors'][i]) == int(row['actor_id'])):
                    raise ValueError('Actor mismatch')
                if not (int(sample['label']) == int(d['labels'][i]) == int(row['label'])):
                    raise ValueError('Label mismatch')
                s, mask, length = sample['F_sequence'], sample['padding_mask'], int(sample['length'])
                if s.ndim != 2 or s.shape[1] != 128 or not np.isfinite(s).all():
                    raise ValueError('Bad sequence')
                if mask.dtype != np.bool_ or mask.shape != (len(s),) or not 1 <= length <= len(s):
                    raise ValueError('Bad mask/length')
                if not np.array_equal(mask,np.arange(len(s)) >= length) or length != int(row['length']):
                    raise ValueError('Padding convention/length mismatch')
                np.testing.assert_allclose(sample['F_audio_learned_feature_only'], d['pooled'][i],atol=1e-6)
                np.testing.assert_allclose(sample['P_audio_learned_feature_only'], d['probs'][i],atol=1e-6)
                lengths.append(length)
        path = destination/f'audio_{split}.npz'
        np.savez_compressed(path,**d)
        load_modality(path)
        converted[split] = path
        sequence_summary[split] = {'count':expected_n,'min_length':min(lengths),'max_length':max(lengths),
                                   'mean_length':float(np.mean(lengths)),
                                   'class_counts':np.bincount(d['labels'],minlength=8).tolist()}
    if seen_actors != set(range(1,25)):
        raise ValueError('Actors do not cover 1..24')
    write_json(destination/'actors_m2.json',actors)
    with open(archive,'rb') as source_file:
        digest = hashlib.file_digest(source_file,'sha256').hexdigest()
    report = {'archive_sha256':digest,
              'metadata':metadata,'actors':actors,'sequences':sequence_summary,
              'class_alias':{'surprised':'surprise'},'all_sample_features_checked':True,
              'test_metrics_computed':False,
              'note':'Adapted split files contain pooled/probs; variable sequences remain in source/per_sample.'}
    write_json(destination/'import_audit.json',report)
    return converted, report


def pilot(archive, destination, output, epochs=20, seed=42):
    started=time.monotonic()
    output=Path(output)
    if (output/'validation_metrics.json').exists():
        raise ValueError('Choose a new output directory to preserve prior results')
    output.mkdir(parents=True,exist_ok=True)
    paths,audit=import_archive(archive,destination)
    # Inspecting test format above is an integrity audit only. No test prediction or metric.
    train,val=load_modality(paths['train']),load_modality(paths['val'])
    seed_all(seed)
    mean=train['pooled'].mean(0)
    std=np.maximum(train['pooled'].std(0),1e-6)
    x=torch.from_numpy((train['pooled']-mean)/std)
    xv=torch.from_numpy((val['pooled']-mean)/std)
    y=torch.as_tensor(train['labels'],dtype=torch.long)
    model=head(128,64,.2)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    history=[]
    best=-1
    best_state=None
    stale=0
    for epoch in range(1,epochs+1):
        model.train()
        loss_sum=0.
        for ix in torch.randperm(len(x)).split(64):
            opt.zero_grad()
            loss=torch.nn.functional.cross_entropy(model(x[ix]),y[ix])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        model.eval()
        with torch.no_grad():
            p=model(xv).softmax(-1).numpy()
        score=metrics(val['labels'],p)['macro_f1']
        history.append({'epoch':epoch,'train_loss':loss_sum/len(x),'val_macro_f1':score})
        if score>best+1e-8:
            best,best_epoch,stale=score,epoch,0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=8:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        new_probs=model(xv).softmax(-1).numpy()
    results={'scope':'Real M2 learned-only audio; validation-only pilot; no visual fusion',
             'seed':seed,'requested_epochs':epochs,'completed_epochs':len(history),'best_epoch':best_epoch,
             'actors':audit['actors'],'test_metrics_computed':False,
             'upstream_audio':metrics(val['labels'],val['probs']),
             'retrained_audio_pooled_mlp':metrics(val['labels'],new_probs),
             'settings':{'input_dim':128,'hidden':64,'dropout':.2,'lr':.001,'batch_size':64,
                         'patience':8,'device':'cpu','frozen_encoder':True},
             'elapsed_seconds':time.monotonic()-started}
    torch.save({'state_dict':best_state,'mean':torch.tensor(mean),'std':torch.tensor(std),
                'class_names':list(CLASSES),'input_dim':128,'hidden':64,'dropout':.2,
                'best_epoch':best_epoch,'seed':seed,'model_type':'audio_only_mlp'},output/'audio_pooled_mlp.pt')
    # Verify the saved checkpoint reproduces the validation prediction.
    ck=torch.load(output/'audio_pooled_mlp.pt',weights_only=True,map_location='cpu')
    restored=head(128,ck['hidden'],ck['dropout']).eval()
    restored.load_state_dict(ck['state_dict'])
    with torch.no_grad():
        restored_probs=restored((torch.from_numpy(val['pooled'])-ck['mean'])/ck['std']).softmax(-1).numpy()
    np.testing.assert_allclose(restored_probs,new_probs,atol=1e-6)
    results['checkpoint_roundtrip_verified']=True
    write_json(output/'validation_metrics.json',results)
    write_json(output/'history.json',history)
    with (output/'validation_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['sample_id','label','model','prediction',*CLASSES])
        for name,probs in [('upstream_audio',val['probs']),('retrained_audio_pooled_mlp',new_probs)]:
            for sid,label,p in zip(val['sample_ids'],val['labels'],probs):
                w.writerow([sid,int(label),name,int(p.argmax()),*p.tolist()])
    print(json.dumps({k:results[k] for k in ['best_epoch','completed_epochs','elapsed_seconds','test_metrics_computed']},indent=2))
    for key in ['upstream_audio','retrained_audio_pooled_mlp']:
        print(key, 'val macro-F1=',results[key]['macro_f1'],'accuracy=',results[key]['accuracy'])
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--data-dir',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.epochs<1:
        p.error('epochs must be positive')
    pilot(args.archive,args.data_dir,args.output,args.epochs,args.seed)
