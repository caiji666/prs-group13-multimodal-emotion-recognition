"""Fixed-budget LMF/SupCon/SVM study and gate-only interventions, validation only."""
import argparse
import copy
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .paths import project_revision
import joblib
import numpy as np
import torch
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from . import CLASSES
from .data import fit_normalizers,normalize
from .experiment import seed_all,infer,tensor_batch,write_json
from .feature_inputs import sha256
from .feature_pilot import val_metrics
from .models import FusionModel
from .structure_study import canonical_metrics,now,diagnostics
from .literature_models import LiteratureModel,METHODS,supervised_contrastive

ROOT=Path(__file__).resolve().parents[2]


def save_predictions(folder,raw,val,rid,pred,prob=None,scores=None):
    with (folder/'predictions_val.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred',*['p_'+c for c in CLASSES]])
        for j,i in enumerate(val):
            w.writerow([rid,'ravdess_speech_v1','val',raw['sample_ids'][i],int(raw['actors'][i]),int(raw['labels'][i]),int(pred[j]),
                        *(map(float,prob[j]) if prob is not None else ['']*8)])
    if scores is not None:
        with (folder/'decision_scores_val.csv').open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['sample_id',*['score_'+c for c in CLASSES]])
            for sid,score in zip(raw['sample_ids'][val],scores):w.writerow([sid,*map(float,score)])


def label_metrics(y,pred):
    # Metrics here depend only on class predictions; these one-hot arrays are never exported as probabilities.
    return canonical_metrics(val_metrics(y,np.eye(8,dtype=np.float32)[pred]))


def record(folder,row,audit,config,started,neural):
    out=folder.parent.parent;r=json.loads((ROOT/'docs/templates/run.template.json').read_text())
    r.update(run_id=row['run_id'],member='M5',owner='Jiang Yushan',stage='fusion',
             comparison_group='old_features_literature_exploration_20261020',pipeline_name=row['method'],run_status='completed',
             test_status='not_run',started_at=started,finished_at=now(),seed=row['seed'],
             split_sha256=audit['source_hashes']['actor_csv'],input_manifest_ref=str(out/'input_manifest.csv'),
             input_manifest_sha256=sha256(out/'input_manifest.csv'),
             git_commit=project_revision(ROOT),
             working_tree_dirty=True,code_patch_ref=str(out/'code'),command=' '.join(sys.argv))
    r['counts']['actual']={'train':960,'val':240,'test':None}
    r['source_runs']['audio_learned']='M2_encoder_transformer_s42_20261015_00'
    r['environment'].update(python=platform.python_version(),framework_and_version=str(torch.__version__) if neural else 'sklearn SVC '+__import__('sklearn').__version__,device='cpu')
    r['input_artifacts']=[dict(source_hashes=audit['source_hashes'],development_cache_sha256=audit['development_cache_sha256'])]
    r['preprocessing'].update(standardization_fit_split='train',augmentation_train_only=False)
    r['pipeline']['feature_extractors'][0].update(component_id='historical_frozen_features',modality='audio+visual',kind='cached',
        method='M1 48D + old MFCC Transformer128D; MobileNet576D',trainable_in_this_run=False)
    r['pipeline']['representation'].update(kind='combined',pooling='supplied pooled',normalization='per-dimension standardization',normalization_fit_split='train',
        components=[dict(modality='audio',dim=176),dict(modality='visual',dim=576)])
    r['pipeline']['fusion'].update(applied=not row['method'].startswith(('svm_audio_','svm_visual_')),method=row['method'],parameters=config,trainable_in_this_run=neural)
    r['pipeline']['classifier'].update(type=('frozen_audio_MLP_plus_residual' if row['method'].startswith('aux_') else 'MLP') if neural else 'SVC',architecture=row['method'],
        hidden_dims=[64] if neural else None,probability_method='softmax' if neural else 'none',trainable_in_this_run=True,
        parameters={'uncalibrated_decision_scores_exported':not neural})
    r['training'].update(regime='frozen_features' if neural else 'fixed_feature_svm',optimizer='AdamW' if neural else 'libsvm',
        learning_rate=.001 if neural else None,batch_size=64 if neural else None,max_epochs=30 if neural else None,
        completed_epochs=row.get('completed_epochs'),early_stopping_patience=8 if neural else None,class_weighting='none',
        other_parameters=dict(configuration=config,plan_sha256=audit['plan_sha256'],
          auxiliary_ce_weight_each=.2 if neural and row['method'].startswith('aux_') else 0,
          contrastive_weight=.1 if neural and 'supcon' in row['method'] else 0))
    r['selection'].update(best_epoch=row.get('best_epoch'),config_locked_at=audit['locked_at'],
        tie_break_rule='earlier epoch; SVM within family: earlier declared candidate on exact tie',
        selection_reason='Validation-only exploration; all candidate records retained')
    r['metrics_by_split']={'val':row['val'],'test':None}
    ck=folder/('checkpoint.pt' if neural else 'checkpoint.joblib')
    r['artifacts'].update(original_metrics_ref=str(folder/'metrics.json'),predictions_val_ref=str(folder/'predictions_val.csv'),
        checkpoint_ref=str(ck),checkpoint_sha256=sha256(ck),history_ref=str(folder/'history.json') if neural else None)
    r['limitations']=['Historical test already exposed; this study only uses train/val.',
        'LMF and SupCon are two-modality task adaptations, not full original-paper replications.',
        'SVM probability columns are empty: decision scores are not calibrated probabilities.',
        'Kernel fusion uses fixed alpha grid, not full SimpleMKL optimization.',
        'All hyperparameter candidates are retained; no test-based selection.']
    r['frozen_audio_reference']=audit['base_references'][str(row['seed'])] if neural and row['method'].startswith('aux_') else None
    (folder/'notes.md').write_text(f'# {row["run_id"]}\n\nOld feature validation-only exploratory experiment. No WavLM or test inference.\n\n'
        'All train standardizers and code/input hashes preserved.\n\n'
        +('Softmax probabilities exported. SupCon mode, weight and temperature are fixed in the plan.\n' if neural else
          'SVC is deterministic with probability=False. Blank probability columns are intentional; separate OVR scores are uncalibrated.\n'))
    write_json(folder/'run.json',r);write_json(folder/'metrics.json',row)


def fit_neural(d,raw,stats,audit,method,seed,base,config,out,audio_pred,visual_pred,prior_aux):
    rid=f'M5_literature_{method}_s{seed}_20261020';folder=out/'runs'/rid;folder.mkdir(parents=True,exist_ok=False)
    seed_all(seed);started=now();clock=time.monotonic();model=LiteratureModel(method,**config)
    if method.startswith('aux_'):model.core.base.load_state_dict(base['state_dict'])
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-4)
    rng=np.random.RandomState(seed);best=-1.;stale=0;history=[]
    for epoch in range(1,31):
        model.train();order=rng.permutation(train);sums=np.zeros(4)
        for j in range(0,len(order),64):
            ix=order[j:j+64];opt.zero_grad();b=tensor_batch(d,ix,'cpu');y=torch.as_tensor(raw['labels'][ix])
            logits,aux,z=model(b,details=True)
            main=torch.nn.functional.cross_entropy(logits,y)
            extra=sum((torch.nn.functional.cross_entropy(p,y) for p in aux),torch.zeros(()))
            contrast=torch.zeros(())
            if method.startswith('aux_supcon'):
                mods=torch.cat((torch.zeros(len(y),dtype=torch.long),torch.ones(len(y),dtype=torch.long)))
                contrast=supervised_contrastive(z,torch.cat((y,y)),mods,mode='intra' if method.endswith('intra') else 'cross',temperature=config['temperature'])
            loss=main+.2*extra+config['contrastive_weight']*contrast
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            sums+=np.array([float(x.detach()) for x in (loss,main,extra,contrast)])*len(ix)
        score=val_metrics(raw['labels'][val],infer(model,d,val,'cpu'))
        history.append(dict(epoch=epoch,train_loss=sums[0]/len(train),train_main_ce=sums[1]/len(train),
            train_aux_ce=sums[2]/len(train),train_supcon=sums[3]/len(train),val_macro_f1=score['macro_f1'],val_accuracy=score['accuracy']))
        if score['macro_f1']>best+1e-8:best=score['macro_f1'];best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(state);p=infer(model,d,val,'cpu');score=canonical_metrics(val_metrics(raw['labels'][val],p))
    if method.startswith('aux_'):
        for k,v in model.core.base.state_dict().items():torch.testing.assert_close(v,base['state_dict'][k],rtol=0,atol=0)
    ck=dict(method=method,config=config,state_dict=state,normalizers=stats,class_names=list(CLASSES),seed=seed,
            best_epoch=best_epoch,source_hashes=audit['source_hashes'],test_status='not_run',plan_sha256=audit['plan_sha256'])
    torch.save(ck,folder/'checkpoint.pt');loaded=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu')
    restored=LiteratureModel(method,**loaded['config']);restored.load_state_dict(loaded['state_dict'])
    np.testing.assert_allclose(infer(restored,d,val,'cpu'),p,rtol=1e-5,atol=1e-6)
    save_predictions(folder,raw,val,rid,p.argmax(1),prob=p);write_json(folder/'history.json',history)
    diag=diagnostics(model,d,val,audio_pred,visual_pred);write_json(folder/'diagnostics_val.json',diag)
    row=dict(run_id=rid,method=method,family='neural',seed=seed,val=score,test=None,test_status='not_run',best_epoch=best_epoch,
        completed_epochs=len(history),folder=str(folder),seconds=time.monotonic()-clock,
        parameters_total=sum(p.numel() for p in model.parameters()),parameters_requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
        checkpoint_roundtrip_verified=True,repairs_audio=diag['repairs_audio'],harms_audio=diag['harms_audio'])
    if method=='aux_control':
        row['historical_aux_f1_difference']=score['macro_f1']-prior_aux['val']['macro_f1']
        assert abs(row['historical_aux_f1_difference'])<1e-12,'Control did not reproduce historical auxiliary model'
    record(folder,row,audit,config,started,True)
    print(f'{method} s{seed}: valF1={score["macro_f1"]:.4f} epoch={best_epoch}/{len(history)}',flush=True)
    return row


def fit_svms(d,raw,stats,audit,out):
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val');y=raw['labels'][train]
    a=d['audio_pooled'];v=d['visual_pooled'];inputs=dict(audio=a,visual=v,concat=np.concatenate((a,v),axis=1))
    # Per-modality RBF kernel diagonals are 1, gamma=1/D after train-only standardization.
    kaa=rbf_kernel(a[train],a[train],gamma=1/a.shape[1]);kva=rbf_kernel(a[val],a[train],gamma=1/a.shape[1])
    kav=rbf_kernel(v[train],v[train],gamma=1/v.shape[1]);kvv=rbf_kernel(v[val],v[train],gamma=1/v.shape[1])
    configs=[]
    for modality in ('audio','visual','concat'):
        for kernel in ('linear','rbf'):
            for c in (.1,1.,10.):configs.append(dict(modality=modality,kernel=kernel,C=c,alpha=None))
    for alpha in (.25,.5,.75):
        for c in (.1,1.,10.):configs.append(dict(modality='kernel_fusion',kernel='precomputed',C=c,alpha=alpha))
    rows=[]
    for i,config in enumerate(configs,1):
        started=now();clock=time.monotonic();m=config['modality'];k=config['kernel'];c=config['C'];alpha=config['alpha']
        method=f'svm_{m}_{k}';rid=f'M5_literature_svm_c{i:02d}_20261020';folder=out/'runs'/rid;folder.mkdir(parents=True,exist_ok=False)
        if k=='precomputed':xt=alpha*kaa+(1-alpha)*kav;xv=alpha*kva+(1-alpha)*kvv;gamma=None
        else:xt=inputs[m][train];xv=inputs[m][val];gamma=1/xt.shape[1]
        clf=SVC(C=c,kernel=k,gamma=gamma if gamma is not None else 'scale',probability=False,
                decision_function_shape='ovr',break_ties=True,cache_size=256)
        clf.fit(xt,y);assert np.array_equal(clf.classes_,np.arange(8))
        pred=clf.predict(xv);scores=clf.decision_function(xv);assert np.array_equal(pred,scores.argmax(1))
        payload=dict(model=clf,normalizers=stats,configuration=config,gamma=gamma,train_sample_ids=raw['sample_ids'][train].tolist(),
                     input_cache_sha256=audit['development_cache_sha256'],test_status='not_run')
        joblib.dump(payload,folder/'checkpoint.joblib',compress=3)
        restored=joblib.load(folder/'checkpoint.joblib')['model'];np.testing.assert_array_equal(restored.predict(xv),pred)
        save_predictions(folder,raw,val,rid,pred,prob=None,scores=scores)
        row=dict(run_id=rid,method=method,family='svm_'+m,seed=None,configuration=dict(config,gamma=gamma),val=label_metrics(raw['labels'][val],pred),
                 test=None,test_status='not_run',folder=str(folder),seconds=time.monotonic()-clock,
                 checkpoint_roundtrip_verified=True,selected_in_family=False,probabilities_available=False)
        record(folder,row,audit,row['configuration'],started,False);rows.append(row)
        print(f'SVM candidate{i:02d} {m} {k} C={c} alpha={alpha}: valF1={row["val"]["macro_f1"]:.4f}',flush=True)
    winners={}
    for family in dict.fromkeys(r['family'] for r in rows):
        winner=max((r for r in rows if r['family']==family),key=lambda r:r['val']['macro_f1']);winner['selected_in_family']=True
        folder=Path(winner['folder']);record_json=json.loads((folder/'run.json').read_text())
        record_json['selection']['selected_for_next_stage']=True;write_json(folder/'run.json',record_json);write_json(folder/'metrics.json',winner)
        winners[family]=winner['run_id']
    write_json(out/'svm_selection.json',dict(rule='Max validation macro-F1 within each predeclared family; exact ties retain earlier candidate',
              total_candidates=len(rows),winners=winners,all_candidates_retained=True,test_status='not_run'))
    return rows


@torch.no_grad()
def gate_logits(model,batch,gate):
    a=torch.tanh(model.audio(batch['audio_pooled']));v=torch.tanh(model.visual(batch['visual_pooled']))
    return model.classifier(gate*a+(1-gate)*v)


def gate_diagnostic(d,raw,prior_rows,out,repeats=20):
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val');y=raw['labels'][val]
    target_folder=out/'gate_diagnostic';target_folder.mkdir();rows=[]
    with torch.no_grad():
        for seed in (42,43,44):
            source=next(r for r in prior_rows if r['method']=='gated' and r['seed']==seed)
            path=Path(source['folder'])/'checkpoint.pt'
            if not path.is_absolute():path=ROOT/'multimodal_fusion'/path
            ck=torch.load(path,map_location='cpu',weights_only=True);model=FusionModel('gated',**ck['config']).eval();model.load_state_dict(ck['state_dict'])
            # Pooled-only full validation batch; frozen features never permuted.
            b={k:torch.as_tensor(d[k][val]) for k in ('audio_pooled','visual_pooled')}
            bt={k:torch.as_tensor(d[k][train]) for k in ('audio_pooled','visual_pooled')}
            g=torch.sigmoid(model.gate(torch.cat((b['audio_pooled'],b['visual_pooled']),-1)))
            train_mean=torch.sigmoid(model.gate(torch.cat((bt['audio_pooled'],bt['visual_pooled']),-1))).mean(0,keepdim=True)
            original=gate_logits(model,b,g).softmax(-1).numpy()
            np.testing.assert_allclose(original,model(b).softmax(-1).numpy(),atol=1e-6,rtol=1e-5)
            assert abs(val_metrics(y,original)['macro_f1']-source['val']['macro_f1'])<1e-12
            pa=gate_logits(model,b,torch.ones_like(g)).softmax(-1).numpy()
            pv=gate_logits(model,b,torch.zeros_like(g)).softmax(-1).numpy()
            ac=pa.argmax(1)==y;vc=pv.argmax(1)==y
            # Label-aware positive control among two extreme gates; never a deployable result.
            choose=np.where(ac!=vc,ac,pa[np.arange(len(y)),y]>=pv[np.arange(len(y)),y])
            oracle=torch.as_tensor(choose,dtype=g.dtype)[:,None].expand_as(g)
            conditions=[('original',None,g),('fixed_half',None,torch.full_like(g,.5)),
                        ('fixed_train_mean',None,train_mean.expand_as(g)),('audio_extreme',None,torch.ones_like(g)),
                        ('visual_extreme',None,torch.zeros_like(g)),('label_oracle_extremes',None,oracle)]
            permutations=[]
            for repeat in range(repeats):
                permutation=np.random.default_rng(20261020+repeat).permutation(len(val));permutations.append(permutation)
                conditions.append(('shuffle_gate',repeat,g[permutation]))
                conditions.append(('shuffle_oracle_positive_control',repeat,oracle[permutation]))
            saved_probs=[]
            for name,repeat,used in conditions:
                q=gate_logits(model,b,used).softmax(-1).numpy();target=original.argmax(1)
                saved_probs.append(q)
                rows.append(dict(seed=seed,condition=name,repeat=repeat,macro_f1=val_metrics(y,q)['macro_f1'],
                    accuracy=float((q.argmax(1)==y).mean()),prediction_change_rate=float((q.argmax(1)!=target).mean()),
                    mean_absolute_probability_change=float(np.abs(q-original).mean()),
                    original_predicted_probability_drop=float((original[np.arange(len(y)),target]-q[np.arange(len(y)),target]).mean()),
                    checkpoint_sha256=sha256(path),uses_labels_for_routing='oracle' in name,test_status='not_run'))
            np.savez_compressed(target_folder/f'gate_values_s{seed}.npz',sample_ids=raw['sample_ids'][val],
                gates=g.numpy(),train_mean=train_mean.numpy(),permutations=np.stack(permutations),label_oracle_gate=oracle.numpy())
            np.savez_compressed(target_folder/f'gate_predictions_s{seed}.npz',sample_ids=raw['sample_ids'][val],labels=y,
                probabilities=np.stack(saved_probs),conditions=np.array([c[0] for c in conditions]),
                repeats=np.array([-1 if c[1] is None else c[1] for c in conditions]))
    write_json(target_folder/'results.json',rows)
    write_json(target_folder/'scope.json',dict(repeats=repeats,inputs_fixed=True,only_gate_overridden=True,
        label_oracle_is_diagnostic_only=True,test_status='not_run',
        caveat='Feature-dependent vector gates are not proven quality scores. Sensitivity is not causal attribution. Oracle only selects between two extreme gates, not a global upper bound.'))
    return rows


def run(pilot,base_study,structure_study,output):
    seed_all(42);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);clock=time.monotonic()
    pilot=Path(pilot).resolve();base_study=Path(base_study).resolve();structure_study=Path(structure_study).resolve()
    with np.load(pilot/'development_features.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    assert len(raw['labels'])==1200 and set(raw['splits'].tolist())=={'train','val'} and not any('prob' in k for k in raw)
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    assert len(train)==960 and len(val)==240 and not(set(raw['actors'][train])&set(raw['actors'][val]))
    stats=fit_normalizers(raw,train);d=normalize(raw,stats)
    audit=json.loads((pilot/'adapter_audit.json').read_text());audit['development_cache_sha256']=sha256(pilot/'development_features.npz')
    assert sha256(ROOT/'multimodal_fusion/configs/actor_splits.csv')==audit['source_hashes']['actor_csv']
    prior=json.loads((base_study/'default_runs.json').read_text());structures=json.loads((structure_study/'runs.json').read_text())
    loaded={};references={};baselines=[]
    for seed in (42,43,44):
        loaded[seed]={}
        for method in ('audio_only','visual_only','concat','gated','cross_attention'):
            r=next(r for r in prior if r['method']==method and r['seed']==seed);path=Path(r['folder'])/'checkpoint.pt'
            if not path.is_absolute():path=ROOT/'multimodal_fusion'/path
            ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['normalizers']==stats and ck['source_hashes']==audit['source_hashes']
            model=FusionModel(method,**ck['config']);model.load_state_dict(ck['state_dict']);p=infer(model,d,val,'cpu')
            score=val_metrics(raw['labels'][val],p);assert abs(score['macro_f1']-r['val']['macro_f1'])<1e-12
            loaded[seed][method]=(ck,p);baselines.append(dict(method=method,seed=seed,val=canonical_metrics(score),checkpoint_sha256=sha256(path)))
            if method=='audio_only':references[str(seed)]=dict(checkpoint_ref=str(path),checkpoint_sha256=sha256(path))
    config=dict(audio_dim=raw['audio_pooled'].shape[-1],visual_dim=raw['visual_pooled'].shape[-1],
        audio_seq_dim=raw['audio_sequence'].shape[-1],visual_seq_dim=raw['visual_sequence'].shape[-1],hidden=64,heads=4,dropout=.2,
        rank=4,temperature=.1,contrastive_weight=.1)
    locked=now();plan=dict(locked_at=locked,methods=list(METHODS),seeds=[42,43,44],config=config,
        epochs=30,patience=8,lr=.001,wd=.0001,batch_size=64,aux_ce_weight_each=.2,uses_wavlm=False,test_status='not_run',
        historical_test_exposed=True,neural_fits=15,svm_fits=27,
        svm_grid=dict(modalities=['audio','visual','concat'],kernels=['linear','rbf'],C=[.1,1.,10.],
                      kernel_fusion_alphas=[.25,.5,.75],gamma='1 / modality dimension after train normalization',probability=False),
        gate_repeats=20,gate_shuffle_seed_base=20261020,selection='Report every config; SVM family chooses max val F1, exact ties retain earlier candidate. No test. No adaptive expansion.',
        baseline_refs=references,source_hashes=audit['source_hashes'],development_cache_sha256=audit['development_cache_sha256'])
    write_json(out/'plan.json',plan);audit.update(plan_sha256=sha256(out/'plan.json'),locked_at=locked,base_references=references)
    write_json(out/'source_audit.json',audit);write_json(out/'reused_baselines.json',baselines)
    shutil.copyfile(pilot/'input_manifest.csv',out/'input_manifest.csv');code=out/'code';code.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,code/p.name)
    write_json(out/'code_hashes.json',{p.name:sha256(p) for p in code.iterdir()})
    gate_diagnostic(d,raw,prior,out,repeats=20)
    rows=[]
    for seed in (42,43,44):
        aux=next(r for r in structures if r['method']=='audio_residual_aux' and r['seed']==seed)
        for method in METHODS:
            rows.append(fit_neural(d,raw,stats,audit,method,seed,loaded[seed]['audio_only'][0],config,out,
                        loaded[seed]['audio_only'][1].argmax(1),loaded[seed]['visual_only'][1].argmax(1),aux))
            write_json(out/'runs.json',rows)
    rows+=fit_svms(d,raw,stats,audit,out);write_json(out/'runs.json',rows)
    write_json(out/'status.json',dict(run_status='completed',new_records=len(rows),neural_fits=15,svm_fits=27,
               gate_interventions=138,uses_wavlm=False,test_status='not_run',seconds=time.monotonic()-clock))
    print('Completed 42 configurations. Validation only.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('pilot','base-study','structure-study','output'):p.add_argument('--'+arg,required=True)
    run(**vars(p.parse_args()))
