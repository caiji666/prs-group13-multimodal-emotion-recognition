"""Validate published M5 v2 records without private features, model weights or retraining."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

CLASSES=['neutral','calm','happy','sad','angry','fearful','disgust','surprise']
FIELDS=['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred',*['p_'+c for c in CLASSES]]


def check_keys(value,template,path=''):
    if isinstance(template,dict):
        assert isinstance(value,dict),path
        assert set(template)<=set(value),(path,set(template)-set(value))
        for k,v in template.items():check_keys(value[k],v,path+'.'+k)
    elif isinstance(template,list) and isinstance(value,list) and template:
        for v in value:check_keys(v,template[0],path+'[]')


def run(root):
    root=Path(root).resolve();records=root/'experiment_records';shared=records/'shared/M5'
    template=json.loads((root/'docs/templates/run.template.json').read_text())
    inventory=json.loads((shared/'export_inventory.json').read_text())
    paths=sorted(records.glob('M5_*/run.json'))
    assert len(paths)==inventory['new_records']==30
    with (root/'multimodal_fusion/configs/actor_splits.csv').open() as f:
        actors={int(r['actor_id']):r['split'] for r in csv.DictReader(f)}
    with (shared/'input_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    lookup={r['sample_id']:r for r in manifest};assert len(manifest)==len(lookup)==1440
    summary=(records/'summary.md').read_text();evaluated=0;tested=0
    for path in paths:
        r=json.loads(path.read_text());check_keys(r,template)
        rid=r['run_id'];assert path.parent.name==rid and r['class_names']==CLASSES
        assert r['source_runs']['audio_learned'] is None or (records/r['source_runs']['audio_learned']/'run.json').exists()
        assert r['test_status']==('completed' if r['metrics_by_split']['test'] is not None else 'not_run')
        for refkey,hashkey in [('split_ref','split_sha256'),('input_manifest_ref','input_manifest_sha256')]:
            p=root/r[refkey];assert p.exists();assert hashlib.sha256(p.read_bytes()).hexdigest()==r[hashkey]
        for split,m in r['metrics_by_split'].items():
            if m is None:
                assert split=='test' and r['artifacts']['predictions_test_ref'] is None
                expected=f'| {rid} | test | — | — | — | — | — | not_run |'
                assert expected in summary
                continue
            with (root/r['artifacts'][f'predictions_{split}_ref']).open() as f:
                reader=csv.DictReader(f);assert reader.fieldnames==FIELDS;rows=list(reader)
            assert len(rows)==m['n']==240
            assert len({x['sample_id'] for x in rows})==240
            for row in rows:
                sid=row['sample_id'];entry=lookup[sid]
                assert row['run_id']==rid and row['split']==entry['split']==split
                assert actors[int(row['actor_id'])]==split and int(row['actor_id'])==int(sid[-2:])
                assert int(row['y_true'])==int(entry['label'])==int(sid.split('-')[1])-1
            y=np.asarray([int(x['y_true']) for x in rows]);pred=np.asarray([int(x['y_pred']) for x in rows])
            p=np.asarray([[float(x['p_'+c]) for c in CLASSES] for x in rows])
            assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5)
            assert np.array_equal(p.argmax(1),pred)
            pr,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
            scores={'accuracy':accuracy_score(y,pred),'macro_precision':pr.mean(),'macro_recall':recall.mean(),'macro_f1':f1.mean()}
            for key,v in scores.items():assert abs(m[key]-v)<1e-12,(rid,split,key)
            assert m['confusion_matrix_counts']==confusion_matrix(y,pred,labels=range(8)).tolist()
            assert support.tolist()==[16]+[32]*7
            assert len(m['per_class'])==8
            for i,v in enumerate(m['per_class']):
                assert v['class_name']==CLASSES[i] and v['label']==i and v['support']==int(support[i])
                for key,values in [('precision',pr),('recall',recall),('f1',f1)]:assert abs(v[key]-values[i])<1e-12
            expected='| '+rid+' | '+split+' | 240 | '+' | '.join(f"{m[k]:.4f}" for k in ['macro_f1','accuracy','macro_precision','macro_recall'])+' | completed |'
            assert expected in summary,(rid,split,'summary mismatch')
            evaluated+=1;tested+=split=='test'
        if r['pipeline']['fusion']['method'] in ('decision_equal','decision_tuned'):
            assert r['pipeline']['feature_extractors']==[]
            assert r['pipeline']['classifier']['type']=='none' and r['pipeline']['classifier']['output_dim'] is None
            assert r['seed'] is None
        history=r['artifacts']['history_ref']
        if history:
            with (root/history).open() as f:rows=list(csv.DictReader(f))
            epoch=r['selection']['best_epoch'];assert epoch in [int(x['epoch']) for x in rows]
            score=float(next(x['val_macro_f1'] for x in rows if int(x['epoch'])==epoch))
            assert abs(score-r['metrics_by_split']['val']['macro_f1'])<1e-12
        for key in ('notes_ref','code_patch_ref'):
            assert (root/r[key]).exists()
        for v in r['source_runs'].values():
            if v:assert (records/v/'run.json').exists(),(rid,v)
    # Confirm existing peer records have not been folded into new-run counts.
    assert (records/inventory['reused_existing_M2_record']/'run.json').exists()
    for p in list(shared.rglob('*'))+[x for path in paths for x in path.parent.iterdir()]:
        if not p.is_file():continue
        assert p.suffix not in ('.pt','.npz','.npy','.pkl','.zip'),p
        if p.suffix in ('.md','.json','.csv','.txt','.py','.patch'):
            text=p.read_text();assert '/Users/jiang/' not in text,p
            assert not re.search(r'gh[pousr]_[A-Za-z0-9]{20,}',text),p
    print(json.dumps({'records':len(paths),'evaluated_prediction_files':evaluated,'completed_test_records':tested,
                      'schema_version':'2.0','metric_summary_history_id_split_checks':'passed'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');run(p.parse_args().root)
