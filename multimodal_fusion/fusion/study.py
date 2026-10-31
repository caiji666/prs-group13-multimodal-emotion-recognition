"""Bounded multi-seed validation study, controls and targeted regularization search."""
import argparse
import copy
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from . import CLASSES
from .data import fit_normalizers, normalize
from .experiment import seed_all, infer, tensor_batch, write_json
from .feature_inputs import sha256
from .feature_pilot import val_metrics
from .models import FusionModel

METHODS=('audio_only','visual_only','concat','gated','cross_attention','audio_sequence_mean','sequence_mean_concat')
FUSIONS=('concat','gated','cross_attention')
from .paths import project_revision

ROOT=Path(__file__).resolve().parents[2]


def write_predictions(path,d,ix,probs):
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','actor_id','split','label','prediction',*CLASSES])
        for i,p in zip(ix,probs):
            w.writerow([d['sample_ids'][i],int(d['actors'][i]),'val',int(d['labels'][i]),int(p.argmax()),*map(float,p)])


def fit_run(d,stats,audit,method,seed,folder,hidden=64,dropout=.2,epochs=30):
    folder=Path(folder)
    if folder.exists():
        raise ValueError('Refusing to overwrite '+str(folder))
    folder.mkdir(parents=True)
    seed_all(seed);started=time.monotonic()
    train=np.flatnonzero(d['splits']=='train');val=np.flatnonzero(d['splits']=='val')
    config=dict(audio_dim=176,visual_dim=576,audio_seq_dim=128,visual_seq_dim=576,hidden=hidden,heads=4,dropout=dropout)
    model=FusionModel(method,**config)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    best=-1;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=np.random.permutation(train);loss_sum=0.
        for j in range(0,len(order),64):
            ix=order[j:j+64];opt.zero_grad()
            logits=model(tensor_batch(d,ix,'cpu'))
            loss=torch.nn.functional.cross_entropy(logits,torch.as_tensor(d['labels'][ix],dtype=torch.long))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            loss_sum+=float(loss.detach())*len(ix)
        score=val_metrics(d['labels'][val],infer(model,d,val,'cpu'))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),val_macro_f1=score['macro_f1'],val_accuracy=score['accuracy']))
        if score['macro_f1']>best+1e-8:
            best=score['macro_f1'];best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(state);probs=infer(model,d,val,'cpu')
    ck=dict(method=method,config=config,state_dict=state,normalizers=stats,class_names=list(CLASSES),
            seed=seed,best_epoch=best_epoch,test_status='not_run',source_hashes=audit['source_hashes'],assumptions=audit['notes'])
    torch.save(ck,folder/'checkpoint.pt')
    loaded=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu')
    restored=FusionModel(method,**loaded['config']);restored.load_state_dict(loaded['state_dict'])
    np.testing.assert_allclose(infer(restored,d,val,'cpu'),probs,atol=1e-6,rtol=1e-5)
    row=dict(method=method,seed=seed,hidden=hidden,dropout=dropout,best_epoch=best_epoch,completed_epochs=len(history),
             val=val_metrics(d['labels'][val],probs),test=None,test_status='not_run',
             parameters=sum(p.numel() for p in model.parameters()),training_seconds=time.monotonic()-started,
             checkpoint_roundtrip_verified=True,folder=str(folder))
    write_json(folder/'metrics.json',row);write_json(folder/'history.json',history)
    write_predictions(folder/'predictions_val.csv',d,val,probs)
    write_run_record(folder,row,config,audit)
    print(f'{method} s{seed} h{hidden} d{dropout}: valF1={best:.4f} best={best_epoch}/{len(history)}',flush=True)
    return row


def write_run_record(folder,row,config,audit):
    template=ROOT/'docs/templates/run.template.json'
    r=json.loads(template.read_text())
    m=row['val'];method=row['method'];seed=row['seed'];fusion=method in FUSIONS
    r.update(run_id=f'M5_{method}_s{seed}_h{row["hidden"]}_d{row["dropout"]}_20261019',member='M5',owner='Jiang Yushan',
             stage='fusion' if fusion else 'control',comparison_group='mobilenet_audio176_validation_study_v1',
             pipeline_name=method,run_status='completed',test_status='not_run',seed=seed,source_hashes=audit['source_hashes'],
             started_at=None,finished_at=datetime.now().astimezone().isoformat(),class_names=list(CLASSES),
             counts={'expected':{'train':960,'val':240,'test':240},'actual':{'train':960,'val':240,'test':None}},
             environment={'python':platform.python_version(),'framework_and_version':str(torch.__version__),'device':'cpu'},
             git_commit=project_revision(ROOT),working_tree_dirty=True,
             code_patch_ref='See study root code/ snapshot',command=' '.join(sys.argv),split_sha256=audit['source_hashes']['actor_csv'])
    r['pipeline']={'feature_extractors':[{'component_id':'audio','trainable_in_this_run':False,'source_run_id':None},
                                       {'component_id':'visual','trainable_in_this_run':False,'source_run_id':None}],
                  'representation':{'audio_pooled_dim':176,'visual_pooled_dim':576,'audio_sequence_dim':128,'visual_sequence_dim':576,
                                    'normalization_fit_split':'train','pooling_control':method if 'mean' in method else None},
                  'fusion':{'applied':fusion or method=='sequence_mean_concat','method':method,'parameters':config},
                  'classifier':{'type':'MLP','hidden_dims':[row['hidden']],'activation':'ReLU','dropout':row['dropout'],
                                'output_dim':8,'probability_method':'softmax','trainable_in_this_run':True}}
    r['training']={'regime':'frozen_encoder_then_classifier','optimizer':'AdamW','learning_rate':.001,'batch_size':64,
                  'max_epochs':30,'completed_epochs':row['completed_epochs'],'early_stopping_patience':8,'class_weighting':'none'}
    r['metrics_by_split']={'val':m,'test':None}
    r['selection']={'metric':'macro_f1','split':'val','best_epoch':row['best_epoch'],'selected_for_next_stage':None,'test_first_evaluated_at':None}
    r['artifacts']={'checkpoint_ref':str(folder/'checkpoint.pt'),'checkpoint_sha256':sha256(folder/'checkpoint.pt'),
                    'predictions_val_ref':str(folder/'predictions_val.csv'),'predictions_test_ref':None,
                    'history_ref':str(folder/'history.json'),'original_metrics_ref':str(folder/'metrics.json')}
    r['limitations']=audit['notes']
    write_json(folder/'run.json',r)


def aggregate(rows):
    summary={}
    for method in METHODS:
        runs=[r for r in rows if r['method']==method]
        if not runs:continue
        values=np.asarray([r['val']['macro_f1'] for r in runs]);acc=np.asarray([r['val']['accuracy'] for r in runs])
        summary[method]={'n':len(runs),'seeds':[r['seed'] for r in runs],'macro_f1_mean':float(values.mean()),
                         'macro_f1_std':float(values.std(ddof=1)) if len(values)>1 else None,
                         'accuracy_mean':float(acc.mean()),'accuracy_std':float(acc.std(ddof=1)) if len(acc)>1 else None}
    return summary


def run(pilot,output):
    pilot=Path(pilot);output=Path(output)
    if output.exists():raise ValueError('Use a fresh study output path')
    output.mkdir(parents=True);started=time.monotonic()
    # Reuse the development-only interface; source test arrays/probs are never reopened.
    with np.load(pilot/'development_features.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    assert len(raw['labels'])==1200 and set(raw['splits'].tolist())=={'train','val'}
    assert not any('prob' in k for k in raw)
    audit=json.loads((pilot/'adapter_audit.json').read_text())
    assert sha256(ROOT/'multimodal_fusion/configs/actor_splits.csv')==audit['source_hashes']['actor_csv']
    train=np.flatnonzero(raw['splits']=='train');stats=fit_normalizers(raw,train);d=normalize(raw,stats)
    manifest=pilot/'input_manifest.csv';shutil.copyfile(manifest,output/'input_manifest.csv')
    write_json(output/'source_audit.json',dict(audit,development_cache_sha256=sha256(pilot/'development_features.npz')))
    plan={'seeds':[42,43,44],'methods':list(METHODS),'default':{'hidden':64,'dropout':.2,'epochs':30,'patience':8},
          'reuse':'Three original seed42 fusion checkpoints after prediction equality check',
          'adaptive_search':'Top two fusion methods by 3-seed mean; seed42 hidden32 or dropout0.4 separately; repeat winning new config on43/44 only if seed42 improves >=0.02.',
          'diagnostics':'Validation only: complementary errors, modality perturbations, gates, attention controls and representative cases.',
          'test_status':'not_run','decision_fusion':'deferred_pending_class_order','excluded_from_scope':'No actor repartition, raw encoders or original test evaluation.'}
    write_json(output/'plan.json',plan)
    code=output/'code';code.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,code/p.name)
    write_json(output/'code_hashes.json',{p.name:sha256(p) for p in code.iterdir()})
    rows=[];val=np.flatnonzero(d['splits']=='val')
    for seed in (42,43,44):
        for method in METHODS:
            if seed==42 and method in FUSIONS:
                source=pilot/method;ck=torch.load(source/'checkpoint.pt',weights_only=True,map_location='cpu')
                assert ck['normalizers']==stats and ck['source_hashes']==audit['source_hashes']
                model=FusionModel(method,**ck['config']);model.load_state_dict(ck['state_dict'])
                p=infer(model,d,val,'cpu');m=val_metrics(d['labels'][val],p)
                old=json.loads((source/'metrics.json').read_text())
                assert m['confusion_matrix']==old['val']['confusion_matrix']
                row=dict(old,method=method,seed=42,hidden=64,dropout=.2,folder=str(source),reused=True)
                print('Reused verified',method,'seed42',flush=True)
            else:row=fit_run(d,stats,audit,method,seed,output/'runs'/f'{method}_s{seed}')
            rows.append(row);write_json(output/'default_runs.json',rows)
    summary=aggregate(rows);write_json(output/'default_summary.json',summary)
    ranked=sorted(FUSIONS,key=lambda m:summary[m]['macro_f1_mean'],reverse=True)[:2]
    tuning=[];chosen=[]
    for method in ranked:
        baseline=next(r for r in rows if r['method']==method and r['seed']==42)
        candidates=[baseline]
        for hidden,dropout in ((32,.2),(64,.4)):
            row=fit_run(d,stats,audit,method,42,output/'tuning'/f'{method}_s42_h{hidden}_d{dropout}',hidden,dropout)
            tuning.append(row);candidates.append(row)
        winner=max(candidates,key=lambda r:r['val']['macro_f1'])
        improvement=winner['val']['macro_f1']-baseline['val']['macro_f1']
        if winner is not baseline and improvement>=.02:
            group=[winner]
            for seed in (43,44):
                row=fit_run(d,stats,audit,method,seed,output/'tuning'/f'{method}_s{seed}_h{winner["hidden"]}_d{winner["dropout"]}',winner['hidden'],winner['dropout'])
                tuning.append(row);group.append(row)
            differences=[g['val']['macro_f1']-next(r for r in rows if r['method']==method and r['seed']==g['seed'])['val']['macro_f1'] for g in group]
            accepted=np.mean(differences)>=.02 and sum(x>0 for x in differences)>=2
            chosen.append({'method':method,'hidden':winner['hidden'],'dropout':winner['dropout'],
                           'differences':differences,'accepted':bool(accepted),'runs':group})
        else:chosen.append({'method':method,'accepted':False,'seed42_improvement':improvement,
                            'reason':'No >=0.02 seed42 gain; do not spend extra seeds on exploratory config'})
        write_json(output/'tuning_runs.json',tuning);write_json(output/'tuning_decisions.json',chosen)
    write_json(output/'study_status.json',{'default_fits':len(rows),'reused_fits':3,'new_fits':len(rows)-3+len(tuning),
               'test_status':'not_run','decision_fusion':'deferred','elapsed_seconds':time.monotonic()-started})
    print(json.dumps(summary,indent=2),flush=True)
    return raw,d,stats,rows,summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.pilot,a.output)
