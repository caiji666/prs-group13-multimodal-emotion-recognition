"""Locked-checkpoint final evaluation; provisional decision fusion with explicit class-order assumption."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from . import CLASSES
from .data import trial_id, normalize
from .experiment import seed_all, infer, write_json
from .feature_inputs import selected_npz_rows, sha256
from .feature_pilot import val_metrics
from .models import FusionModel

ROOT=Path(__file__).resolve().parents[2]


def read_actor_map(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    mapping={int(r['actor_id']):r['split'] for r in rows}
    if len(rows)!=24 or set(mapping)!=set(range(1,25)):
        raise ValueError('Expected 24 unique actors')
    return mapping


def load_test_features(audio_csv,audio_sequences,visual_npz,actor_csv,expected_n=240):
    mapping=read_actor_map(actor_csv)
    with np.load(visual_npz,allow_pickle=False) as z:
        all_ids,all_splits=z['sample_id'],z['split']
    if len(set(all_ids.tolist()))!=len(all_ids):raise ValueError('Duplicate visual IDs')
    ix=np.flatnonzero(all_splits=='test');ids=all_ids[ix]
    if len(ids)!=expected_n:raise ValueError('Unexpected test count')
    labels=selected_npz_rows(visual_npz,'label',ix)
    pooled=selected_npz_rows(visual_npz,'pooled_features',ix)
    sequence=selected_npz_rows(visual_npz,'sequence_features',ix)
    if pooled.shape!=(expected_n,576) or sequence.shape!=(expected_n,16,576):
        raise ValueError('Unexpected visual feature shape')
    visual={str(s):(j,int(l)) for j,(s,l) in enumerate(zip(ids,labels))}
    audio={}
    with Path(audio_csv).open() as f:
        reader=csv.DictReader(f)
        metadata={'sample_id','trial_id','actor_id','emotion_label','split'}
        keys=[k for k in reader.fieldnames if k not in metadata]
        if len(keys)!=176:raise ValueError('Expected 176D audio')
        for r in reader:
            if r['split']!='test':continue
            sid=trial_id(r['sample_id']);actor=int(sid[-2:])
            lab=CLASSES.index('surprise' if r['emotion_label']=='surprised' else r['emotion_label'])
            if mapping[actor]!='test' or int(r['actor_id'])!=actor or int(sid.split('-')[1])-1!=lab:
                raise ValueError('Audio test metadata mismatch')
            if sid in audio:raise ValueError('Duplicate audio ID')
            audio[sid]=(np.asarray([float(r[k]) for k in keys],np.float32),lab)
    if set(audio)!=set(visual):raise ValueError('Test pair ID mismatch')
    ids=sorted(audio);seqs=[];lengths=[]
    for sid in ids:
        j,lab=visual[sid]
        if trial_id('03-'+sid+'.wav')!=sid or mapping[int(sid[-2:])]!='test' or lab!=audio[sid][1]:
            raise ValueError('Visual metadata mismatch')
        with np.load(Path(audio_sequences)/'test'/f'{sid}.npz',allow_pickle=False) as z:
            seq=z['F_sequence'];length=int(z['length']);mask=z['padding_mask']
            if (str(z['sample_id']),str(z['split']),int(z['label']),int(z['actor_id']))!=(sid,'test',lab,int(sid[-2:])):
                raise ValueError('Sequence metadata mismatch')
            if seq.shape!=(length,128) or length<1 or mask.dtype!=np.bool_ or mask.shape!=(length,) or mask.any():
                raise ValueError('Bad test sequence')
            if not np.allclose(seq.mean(0),audio[sid][0][48:],atol=1e-5):
                raise ValueError('Sequence/pooled mismatch')
            seqs.append(seq.copy());lengths.append(length)
    padded=np.zeros((len(ids),max(lengths),128),np.float32)
    for i,s in enumerate(seqs):padded[i,:len(s)]=s
    order=np.asarray([visual[s][0] for s in ids])
    d={'sample_ids':np.asarray(ids),'labels':np.asarray([audio[s][1] for s in ids],np.int64),
       'actors':np.asarray([int(s[-2:]) for s in ids],np.int64),'splits':np.full(len(ids),'test'),
       'audio_pooled':np.stack([audio[s][0] for s in ids]),'visual_pooled':pooled[order],
       'audio_sequence':padded,'visual_sequence':sequence[order],
       'audio_lengths':np.asarray(lengths,np.int64),'visual_lengths':np.full(len(ids),16,np.int64)}
    for k in ('audio_pooled','visual_pooled','audio_sequence','visual_sequence'):
        if not np.isfinite(d[k]).all():raise ValueError('Nonfinite test input '+k)
    if set(d['labels'].tolist())!=set(range(8)):raise ValueError('Missing test class')
    return d


def load_probabilities(visual_npz,audio_dir,split,actor_csv,expected_n=240):
    mapping=read_actor_map(actor_csv)
    with np.load(visual_npz,allow_pickle=False) as z:ids=z['sample_id'];splits=z['split']
    ix=np.flatnonzero(splits==split)
    vp=selected_npz_rows(visual_npz,'probabilities',ix)
    labels=selected_npz_rows(visual_npz,'label',ix)
    visual={str(s):(int(l),p) for s,l,p in zip(ids[ix],labels,vp)}
    audio={}
    with (Path(audio_dir)/f'audio_probabilities_{split}.csv').open() as f:
        reader=csv.DictReader(f)
        keys=[k for k in reader.fieldnames if k.startswith('prob_')]
        if keys!=['prob_'+c for c in CLASSES]:raise ValueError('Unexpected audio probability order')
        for r in reader:
            sid=trial_id(r['sample_id']);lab=CLASSES.index('surprise' if r['emotion_label']=='surprised' else r['emotion_label'])
            if r['split']!=split or mapping[int(sid[-2:])]!=split or lab!=int(sid.split('-')[1])-1:
                raise ValueError('Probability metadata mismatch')
            if sid in audio:raise ValueError('Duplicate probability ID')
            audio[sid]=(lab,np.asarray([float(r[k]) for k in keys]))
    if len(audio)!=expected_n or set(audio)!=set(visual):raise ValueError('Probability pair mismatch')
    ids=sorted(audio)
    if any(audio[s][0]!=visual[s][0] for s in ids):raise ValueError('Probability label mismatch')
    pa=np.stack([audio[s][1] for s in ids]);pv=np.stack([visual[s][1] for s in ids])
    for p in (pa,pv):
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
            raise ValueError('Invalid probabilities')
    return np.asarray(ids),np.asarray([audio[s][0] for s in ids]),pa,pv


def select_alpha(y,a,v):
    grid=[{'alpha_audio':float(alpha),'val_macro_f1':val_metrics(y,alpha*a+(1-alpha)*v)['macro_f1']}
          for alpha in np.linspace(0,1,21)]
    winner=max(grid,key=lambda r:(r['val_macro_f1'],-abs(r['alpha_audio']-.5)))
    return winner['alpha_audio'],grid


def save_predictions(path,ids,labels,split,probs):
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','actor_id','split','label','prediction',*CLASSES])
        for sid,l,p in zip(ids,labels,probs):w.writerow([sid,int(sid[-2:]),split,int(l),int(p.argmax()),*map(float,p)])


def metric_record(m):
    return {'n':m['n'],'accuracy':m['accuracy'],'macro_precision':m['macro_precision'],
            'macro_recall':m['macro_recall'],'macro_f1':m['macro_f1'],
            'per_class':[{'label':i,'class_name':c,'precision':m['per_class'][c]['precision'],
                         'recall':m['per_class'][c]['recall'],'f1':m['per_class'][c]['f1-score'],
                         'support':int(m['per_class'][c]['support'])} for i,c in enumerate(CLASSES)],
            'confusion_matrix_counts':m['confusion_matrix']}


def run(study,output):
    study=Path(study);output=Path(output)
    if output.exists():raise ValueError('Use a new output directory')
    output.mkdir(parents=True);seed_all(0)
    rows=json.loads((study/'default_runs.json').read_text())
    summaries=json.loads((study/'default_summary.json').read_text())
    source=json.loads((study/'source_audit.json').read_text())
    actor_csv=ROOT/'multimodal_fusion/configs/actor_splits.csv'
    audio_csv=ROOT/'features/Audio_Handoff/features/audio_fused_176d.csv'
    visual_npz=ROOT/'features/mobilenetv3_small_features.npz'
    audio_dir=ROOT/'features/Audio_Handoff/probabilities'
    sequence_dir=ROOT/'features/Audio_dl_feature_only/per_sample'
    for key,path in [('actor_csv',actor_csv),('audio_csv',audio_csv),('visual_npz',visual_npz)]:
        if sha256(path)!=source['source_hashes'][key]:raise ValueError('Upstream features changed '+key)
    primary=max(('concat','gated','cross_attention'),key=lambda m:summaries[m]['macro_f1_mean'])
    baseline=max(('audio_only','visual_only'),key=lambda m:summaries[m]['macro_f1_mean'])
    locked=[]
    for row in rows:
        folder=Path(row['folder'])
        if not folder.exists():folder=ROOT/'multimodal_fusion'/folder
        ck_path=folder/'checkpoint.pt';ck=torch.load(ck_path,weights_only=True,map_location='cpu')
        if ck['config']['hidden']!=64 or ck['config']['dropout']!=.2 or ck['seed']!=row['seed']:
            raise ValueError('Unexpected configuration')
        locked.append({'method':row['method'],'seed':row['seed'],'best_epoch':ck['best_epoch'],
                       'checkpoint':str(ck_path),'checkpoint_sha256':sha256(ck_path)})
    # No test features/probabilities have been loaded. Lock candidates and decision search rules first.
    plan={'locked_at':datetime.now().astimezone().isoformat(),'primary_method_selected_by_validation':primary,
          'strongest_matched_unimodal_selected_by_validation':baseline,'candidates':locked,
          'excluded':'Four exploratory single-seed tuning runs not selected by validation policy',
          'decision_search':{'split':'val','alpha_audio_grid':[float(x) for x in np.linspace(0,1,21)],'tie_break':'closest to 0.5'},
          'visual_probability_order_status':'unconfirmed_user_authorized_provisional',
          'assumed_visual_probability_order':list(CLASSES),'no_probability_permutation_search':True,
          'no_training_or_scaler_refitting':True,'no_reselection_from_test':True}
    write_json(output/'locked_plan.json',plan)
    # User explicitly authorizes trying the agreed order provisionally; do not infer it from scores.
    vids,vy,va,vv=load_probabilities(visual_npz,audio_dir,'val',actor_csv)
    alpha,grid=select_alpha(vy,va,vv)
    write_json(output/'decision_validation_grid.json',grid)
    write_json(output/'locked_decision.json',{'locked_before_test':True,'alpha_audio':alpha,'selected_using':'val_macro_f1',
              'visual_class_order_confirmed':False,'assumed_order':list(CLASSES),'provisional':True})
    result={'primary_method_selected_by_validation':primary,'baseline_selected_by_validation':baseline,
            'models':[],'decision':{},'class_names':list(CLASSES),'test_status':'completed','no_retraining':True,
            'provisional_decision_order':True,'test_started_at':datetime.now().astimezone().isoformat()}
    raw=load_test_features(audio_csv,sequence_dir,visual_npz,actor_csv)
    with (output/'test_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','actor_id','split','label'])
        w.writerows(zip(raw['sample_ids'],raw['actors'],raw['splits'],raw['labels']))
    for row,lock in zip(rows,locked):
        ck=torch.load(lock['checkpoint'],weights_only=True,map_location='cpu')
        if sha256(lock['checkpoint'])!=lock['checkpoint_sha256']:raise ValueError('Checkpoint changed after locking')
        d=normalize(raw,ck['normalizers'])
        model=FusionModel(ck['method'],**ck['config']).eval();model.load_state_dict(ck['state_dict'])
        probs=infer(model,d,np.arange(len(raw['labels'])),'cpu')
        m=val_metrics(raw['labels'],probs)
        folder=output/f'{row["method"]}_s{row["seed"]}';folder.mkdir()
        save_predictions(folder/'predictions_test.csv',raw['sample_ids'],raw['labels'],'test',probs)
        record=dict(method=row['method'],seed=row['seed'],val=row['val'],test=m,checkpoint=lock['checkpoint'],
                    checkpoint_sha256=lock['checkpoint_sha256'],best_epoch=ck['best_epoch'],test_status='completed')
        write_json(folder/'metrics.json',record);result['models'].append(record)
        # Extend the same upstream run record; checkpoint weights/history remain unchanged.
        source_run=Path(lock['checkpoint']).parent/'run.json'
        run_record=json.loads(source_run.read_text())
        run_record['test_status']='completed';run_record['metrics_by_split']['test']=metric_record(m)
        run_record['counts']['actual']['test']=240
        run_record['selection']['test_first_evaluated_at']=result['test_started_at']
        run_record['artifacts']['predictions_test_ref']=str(folder/'predictions_test.csv')
        write_json(folder/'run.json',run_record);write_json(source_run,run_record)
        write_json(output/'results.json',result)
        print(f'{row["method"]} s{row["seed"]}: test F1={m["macro_f1"]:.4f}',flush=True)
    tids,ty,ta,tv=load_probabilities(visual_npz,audio_dir,'test',actor_csv)
    if not np.array_equal(tids,raw['sample_ids']) or not np.array_equal(ty,raw['labels']):
        raise ValueError('Test probability/feature alignment mismatch')
    for name,weight in [('upstream_audio',1.),('upstream_visual_assumed_order',0.),('decision_equal',.5),('decision_tuned',alpha)]:
        valp=weight*va+(1-weight)*vv;testp=weight*ta+(1-weight)*tv
        provisional=name!='upstream_audio'
        folder=output/name;folder.mkdir()
        record={'alpha_audio':weight,'val':val_metrics(vy,valp),'test':val_metrics(ty,testp),
                'provisional_unconfirmed_visual_class_order':provisional,'assumed_class_names':list(CLASSES),
                'test_status':'completed','weight_selection':'val only'}
        write_json(folder/'metrics.json',record)
        save_predictions(folder/'predictions_val.csv',vids,vy,'val',valp)
        save_predictions(folder/'predictions_test.csv',tids,ty,'test',testp)
        result['decision'][name]=record
    summary={}
    for method in summaries:
        rr=[r for r in result['models'] if r['method']==method]
        for split in ('val','test'):
            f1=np.array([r[split]['macro_f1'] for r in rr]);acc=np.array([r[split]['accuracy'] for r in rr])
            summary.setdefault(method,{})[split]={'n':len(rr),'macro_f1_mean':float(f1.mean()),'macro_f1_std':float(f1.std(ddof=1)),
                                                    'accuracy_mean':float(acc.mean()),'accuracy_std':float(acc.std(ddof=1))}
    result['summary']=summary;result['test_finished_at']=datetime.now().astimezone().isoformat()
    result['primary_test_delta_vs_matched_baseline']=summary[primary]['test']['macro_f1_mean']-summary[baseline]['test']['macro_f1_mean']
    write_json(output/'results.json',result)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','n','val_macro_f1_mean','test_macro_f1_mean','test_macro_f1_std','test_accuracy_mean','provisional'])
        for method,s in summary.items():w.writerow([method,3,s['val']['macro_f1_mean'],s['test']['macro_f1_mean'],s['test']['macro_f1_std'],s['test']['accuracy_mean'],False])
        for method,r in result['decision'].items():w.writerow([method,1,r['val']['macro_f1'],r['test']['macro_f1'],None,r['test']['accuracy'],r['provisional_unconfirmed_visual_class_order']])
    print('Locked final evaluation complete; decision results remain provisional.',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.study,a.output)
