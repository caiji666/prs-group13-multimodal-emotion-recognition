"""Locked full replay on final audio/visual deliveries; test read only after validation lock."""
import argparse,copy,csv,json,platform,shutil,subprocess,time
from pathlib import Path
from .paths import project_revision
import joblib,numpy as np,torch
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from . import CLASSES
from .data import fit_normalizers,normalize
from .experiment import seed_all,infer,tensor_batch,write_json
from .feature_inputs import sha256
from .feature_pilot import val_metrics
from .final_inputs import FinalInputs,view
from .final_models import FinalModel,DeliveredAudioHead
from .structure_models import METHODS as STRUCTURES
from .literature_models import METHODS as LITERATURE,supervised_contrastive
from .structure_study import canonical_metrics,now,diagnostics
from .literature_study import gate_diagnostic

ROOT=Path(__file__).resolve().parents[2]
BASE=('audio_only','visual_only','concat','gated','cross_attention','audio_sequence_mean','sequence_mean_concat')


def metrics(y,p):return canonical_metrics(val_metrics(y,p))


def save_predictions(folder,raw,ix,rid,p,split,pred=None,scores=None):
    prediction=p.argmax(1) if p is not None else pred
    with (folder/f'predictions_{split}.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred',*['p_'+c for c in CLASSES]])
        for j,i in enumerate(ix):w.writerow([rid,'ravdess_speech_v1',split,raw['sample_ids'][i],int(raw['actors'][i]),int(raw['labels'][i]),int(prediction[j]),*(map(float,p[j]) if p is not None else ['']*8)])
    if scores is not None:
        with (folder/f'decision_scores_{split}.csv').open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['sample_id',*['score_'+c for c in CLASSES]])
            for sid,s in zip(raw['sample_ids'][ix],scores):w.writerow([sid,*map(float,s)])


def record(out,row,audit,config,history=None):
    folder=Path(row['folder']);r=json.loads((ROOT/'docs/templates/run.template.json').read_text())
    rel=lambda p:str(Path(p).relative_to(ROOT))
    r.update(run_id=row['run_id'],member='M5',owner='Jiang Yushan',stage='fusion',comparison_group='final_delivery_20261027',
       pipeline_name=row['method'],run_status='completed',test_status='not_run',started_at=row['started'],finished_at=now(),seed=row['seed'],
       split_sha256=audit['source_hashes']['actor_csv'],input_manifest_ref=rel(out/'input_manifest.csv'),input_manifest_sha256=sha256(out/'input_manifest.csv'),
       git_commit=project_revision(ROOT),working_tree_dirty=True,code_patch_ref=rel(out/'code'),command='python -m fusion.final_study')
    r['counts']['actual']={'train':960,'val':240,'test':None};r['excluded_samples'].update(count=0,handling_policy='All matched recordings retained')
    r['environment'].update(python=platform.python_version(),framework_and_version=str(torch.__version__),device='cpu')
    r['source_runs']['audio_learned']=audit['audio_metadata']['selection_run_id']
    r['input_artifacts']=[dict(role=k,ref=None,sha256=v,source_run_id=None) for k,v in audit['source_hashes'].items()]
    r['preprocessing'].update(standardization_fit_split='train',augmentation_train_only=True,parameters={'physical_alignment':'unknown audio trim offset; relative token positions only'})
    r['pipeline']['feature_extractors'][0].update(component_id='audio',modality='audio',kind='learned',method=audit['audio_metadata']['model'],source_run_id=r['source_runs']['audio_learned'],checkpoint_sha256=audit['audio_metadata']['checkpoint_sha256'],trainable_in_this_run=False,output_pooled_shape=[None,128],output_sequence_shape=[None,None,128])
    v=copy.deepcopy(json.loads((ROOT/'docs/templates/run.template.json').read_text())['pipeline']['feature_extractors'][0])
    v.update(component_id='visual',modality='visual',kind='handcrafted' if row['visual']=='hc' else row['visual'],method='M3 supplied '+row['visual'],trainable_in_this_run=False,
             output_pooled_shape=[None,{'hc':320,'deep':576,'combined':896}[row['visual']]])
    r['pipeline']['feature_extractors'].append(v)
    r['pipeline']['representation'].update(kind='combined',normalization='per dimension train standardization',normalization_fit_split='train',pooling='supplied WavLM attentive statistics; visual mean',
       components=[{'audio':row['audio'],'visual':row['visual']}])
    decision=row['family']=='reference';svm=row['family']=='svm'
    r['pipeline']['fusion'].update(applied=row['method'] not in ('audio_only','visual_only','audio_sequence_mean','upstream_audio','upstream_visual'),method=row['method'],parameters=config,trainable_in_this_run=not decision)
    r['pipeline']['classifier'].update(type='none' if decision else 'SVC' if svm else 'MLP',architecture=row['method'],input_dim=config.get('audio_dim'),hidden_dims=None if decision or svm else [64],
       output_dim=None if decision else 8,activation=None if decision or svm else 'ReLU',dropout=None if decision or svm else .2,probability_method='none' if decision or svm else 'softmax',trainable_in_this_run=not decision)
    r['training'].update(regime='probability_fusion' if decision else 'frozen_encoder_then_classifier',optimizer=None if decision else 'libsvm' if svm else 'AdamW',learning_rate=None if decision or svm else .001,
       batch_size=None if decision or svm else 64,max_epochs=None if decision or svm else 30,completed_epochs=len(history) if history else None,early_stopping_patience=None if decision or svm else 8,class_weighting='none',other_parameters={'plan_sha256':audit['plan_sha256'],'configuration':config})
    r['selection'].update(best_epoch=row.get('best_epoch'),config_locked_at=audit['plan_locked_at'],selection_reason='Validation selection before new test inference; historical test already exposed',tie_break_rule='earlier epoch; decision weight nearest .5 on F1 tie')
    r['metrics_by_split']={'val':row['val'],'test':None}
    checkpoint=folder/('checkpoint.joblib' if svm else 'checkpoint.pt')
    if checkpoint.exists():r['artifacts'].update(checkpoint_ref=rel(checkpoint),checkpoint_sha256=sha256(checkpoint))
    r['artifacts'].update(predictions_val_ref=rel(folder/'predictions_val.csv'),predictions_test_ref=None,history_ref=rel(folder/'history.json') if history else None,original_metrics_ref=rel(folder/'metrics.json'))
    r['final_input_choices']={'audio':row['audio'],'visual':row['visual'],'family':row['family']}
    r['limitations']=['Final input is WavLM-only plus selected handcrafted visual; alternatives explicitly labelled.',
       'No train probability stacking: audio train probabilities are in-sample; M3 train probabilities supplied OOF.',
       'Historical tests were exposed; complete replay is exploratory, not a fresh independent test.',
       'Relative sequence positions are not verified physical alignment. Upstream visual source/run ID not supplied.']
    write_json(folder/'run.json',r);write_json(folder/'metrics.json',row)


def fit(out,raw,d,stats,audit,family,method,visual,audio,seed,source):
    rid=f'M5_final_{family}_{method}_{audio}_{visual}_s{seed}_20261027';folder=out/'runs'/rid;folder.mkdir(parents=True,exist_ok=False)
    seed_all(seed);started=now();train=np.flatnonzero(d['splits']=='train');val=np.flatnonzero(d['splits']=='val')
    config=dict(audio_dim=d['audio_pooled'].shape[-1],visual_dim=d['visual_pooled'].shape[-1],audio_seq_dim=128,visual_seq_dim=d['visual_sequence'].shape[-1],hidden=64,heads=4,dropout=.2)
    if family=='structure':config.update(modality_dropout=.25,aux_weight=.2)
    if family=='literature':config.update(rank=4,temperature=.1,contrastive_weight=.1)
    model=FinalModel(family,method,config,stats,source)
    base=model.inner.core.base if family=='literature' and method.startswith('aux_') else getattr(model.inner,'base',None)
    if base is not None:
        base.eval();np.testing.assert_allclose(infer(base,d,val,'cpu'),raw['audio_probs'][val],atol=2e-6,rtol=1e-4)
        frozen=copy.deepcopy(base.state_dict())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-4)
    best=-1.;stale=0;history=[];rng=np.random.RandomState(seed)
    for epoch in range(1,31):
        model.train();order=rng.permutation(train);total=0.
        for j in range(0,len(order),64):
            ix=order[j:j+64];optimizer.zero_grad();y=torch.as_tensor(d['labels'][ix]);logits,aux,z=model.training_outputs(tensor_batch(d,ix,'cpu'))
            loss=torch.nn.functional.cross_entropy(logits,y)+.2*sum((torch.nn.functional.cross_entropy(p,y) for p in aux),torch.zeros(()))
            if method.startswith('aux_supcon'):
                mods=torch.cat((torch.zeros(len(y),dtype=torch.long),torch.ones(len(y),dtype=torch.long)))
                loss=loss+.1*supervised_contrastive(z,torch.cat((y,y)),mods,'intra' if method.endswith('intra') else 'cross',.1)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(ix)
        score=metrics(d['labels'][val],infer(model,d,val,'cpu'));history.append(dict(epoch=epoch,train_loss=total/len(train),val_macro_f1=score['macro_f1'],val_accuracy=score['accuracy']))
        if score['macro_f1']>best+1e-8:best=score['macro_f1'];best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=8:break
    model.load_state_dict(state);p=infer(model,d,val,'cpu')
    if base is not None:
        for k,v in base.state_dict().items():torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)
    ck=dict(family=family,method=method,config=config,state_dict=state,normalizers=stats,visual=visual,audio=audio,seed=seed,best_epoch=best_epoch,
      source_hashes=audit['source_hashes'],plan_sha256=audit['plan_sha256'],test_status='not_run',class_names=list(CLASSES))
    torch.save(ck,folder/'checkpoint.pt');restored=FinalModel(family,method,config,stats);restored.load_state_dict(torch.load(folder/'checkpoint.pt',weights_only=True)['state_dict'])
    np.testing.assert_allclose(infer(restored,d,val,'cpu'),p,atol=1e-6,rtol=1e-5)
    row=dict(run_id=rid,folder=str(folder),started=started,family=family,method=method,visual=visual,audio=audio,seed=seed,val=metrics(d['labels'][val],p),test=None,
      test_status='not_run',best_epoch=best_epoch,parameters=sum(p.numel() for p in model.parameters()),checkpoint_roundtrip_verified=True)
    save_predictions(folder,raw,val,rid,p,'val');write_json(folder/'history.json',history)
    if visual=='hc' and audio=='learned':write_json(folder/'diagnostics_val.json',diagnostics(model,d,val,raw['audio_probs'][val].argmax(1),raw['visual_hc_probs'][val].argmax(1)))
    record(out,row,audit,config,history)
    if family=='base' and method=='gated' and visual=='hc':
        proxy=out/'gate_sources'/str(seed);proxy.mkdir(parents=True);torch.save(dict(method='gated',config=config,state_dict=model.inner.state_dict()),proxy/'checkpoint.pt')
    print(f'{family}/{method} A={audio} V={visual} s{seed}: valF1={best:.4f} epoch={best_epoch}/{len(history)}',flush=True)
    return row


def svms(out,raw,d,stats,audit):
    train=np.flatnonzero(d['splits']=='train');val=np.flatnonzero(d['splits']=='val');a=d['audio_pooled'];v=d['visual_pooled'];x={'audio':a,'visual':v,'concat':np.concatenate((a,v),1)}
    ka=rbf_kernel(a[train],a[train],gamma=1/a.shape[1]);kav=rbf_kernel(a[val],a[train],gamma=1/a.shape[1])
    kv=rbf_kernel(v[train],v[train],gamma=1/v.shape[1]);kvv=rbf_kernel(v[val],v[train],gamma=1/v.shape[1])
    configs=[dict(modality=m,kernel=k,C=c,alpha=None) for m in x for k in ('linear','rbf') for c in (.1,1.,10.)]+[dict(modality='kernel_fusion',kernel='precomputed',C=c,alpha=w) for w in (.25,.5,.75) for c in (.1,1.,10.)]
    rows=[]
    for i,cfg in enumerate(configs,1):
        rid=f'M5_final_svm_c{i:02d}_20261027';folder=out/'runs'/rid;folder.mkdir();started=now();m=cfg['modality'];k=cfg['kernel'];w=cfg['alpha']
        if k=='precomputed':xt=w*ka+(1-w)*kv;xv=w*kav+(1-w)*kvv;gamma='scale'
        else:xt=x[m][train];xv=x[m][val];gamma=1/xt.shape[1]
        clf=SVC(C=cfg['C'],kernel=k,gamma=gamma,probability=False,decision_function_shape='ovr',break_ties=True).fit(xt,d['labels'][train])
        pred=clf.predict(xv);scores=clf.decision_function(xv);assert np.array_equal(pred,scores.argmax(1))
        joblib.dump(dict(model=clf,config=cfg,normalizers=stats,gamma=gamma,train_ids=raw['sample_ids'][train].tolist()),folder/'checkpoint.joblib',compress=3)
        np.testing.assert_array_equal(joblib.load(folder/'checkpoint.joblib')['model'].predict(xv),pred)
        row=dict(run_id=rid,folder=str(folder),started=started,family='svm',method='svm_'+m,visual='hc',audio='learned',seed=None,
          val=metrics(d['labels'][val],np.eye(8)[pred]),test=None,test_status='not_run',configuration=cfg,checkpoint_roundtrip_verified=True)
        save_predictions(folder,raw,val,rid,None,'val',pred,scores);record(out,row,audit,cfg);rows.append(row)
    return rows


def references(out,raw,audit):
    ix=np.flatnonzero(raw['splits']=='val');y=raw['labels'][ix];rows=[];grid={}
    candidates=[('upstream_audio','hc',1.)]+[('upstream_visual',v,0.) for v in ('hc','deep','combined')]
    for v in ('hc','deep','combined'):
        search=[dict(alpha=float(a),macro_f1=metrics(y,a*raw['audio_probs'][ix]+(1-a)*raw[f'visual_{v}_probs'][ix])['macro_f1']) for a in np.linspace(0,1,21)]
        winner=max(search,key=lambda r:(r['macro_f1'],-abs(r['alpha']-.5)));grid[v]=dict(grid=search,selected=winner)
        candidates.extend([('decision_equal',v,.5),('decision_tuned',v,winner['alpha'])])
    for method,v,alpha in candidates:
        rid=f'M5_final_{method}_{v}_sNA_20261027';folder=out/'runs'/rid;folder.mkdir(parents=True)
        p=alpha*raw['audio_probs'][ix]+(1-alpha)*raw[f'visual_{v}_probs'][ix]
        row=dict(run_id=rid,folder=str(folder),started=now(),family='reference',method=method,visual=v,audio='learned',seed=None,alpha=alpha,val=metrics(y,p),test=None,test_status='not_run')
        save_predictions(folder,raw,ix,rid,p,'val');record(out,row,audit,{'alpha_audio':alpha,'train_probability_stacking':False});rows.append(row)
    write_json(out/'decision_validation_grid.json',grid);return rows


def run(audio,visual,actors,handcrafted,source_checkpoint,development,output):
    seed_all(42);out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    source=torch.load(source_checkpoint,weights_only=True,map_location='cpu');reader=FinalInputs(audio,visual,actors,handcrafted)
    assert sha256(source_checkpoint)==reader.meta['checkpoint_sha256']
    with np.load(Path(development)/'development.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    audit=json.loads((Path(development)/'development_audit.json').read_text());assert audit['source_hashes']==reader.hashes and audit['test_numeric_rows_loaded']==0
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    d0=view(raw);stats0=fit_normalizers(d0,train);dn=normalize(d0,stats0);head=DeliveredAudioHead(stats0)
    head.classifier.load_state_dict({k:source['state_dict']['head.'+k] for k in head.classifier.state_dict()});head.eval()
    np.testing.assert_allclose(infer(head,dn,np.arange(len(raw['labels'])),'cpu'),raw['audio_probs'],atol=2e-6,rtol=1e-4)
    plan=dict(locked_at=now(),seeds=[42,43,44],primary_audio='WavLM-only128',primary_visual='handcrafted320',
      primary_fusions=['concat','gated','cross_attention'],base_methods=list(BASE),structure_methods=list(STRUCTURES),literature_methods=list(LITERATURE),
      representation_ablation=dict(visual=['deep','combined'],methods=['visual_only','concat','gated','cross_attention']),audio_ablation=['handcrafted','combined'],
      neural_fits=93,svm_fits=27,fixed_probability_pipelines=10,max_epochs=30,patience=8,lr=.001,wd=.0001,batch=64,hidden=64,dropout=.2,
      source_hashes=reader.hashes,source_audio_checkpoint_sha256=sha256(source_checkpoint),
      selection='Primary fusion by 3seed mean validation F1 among selected-input concat/gated/cross. Baseline original delivered audio head. All predeclared candidates evaluated once on test after hashes locked; no test reselection.',
      svm_budget='audio/visual/concat x linear/rbf x C .1,1,10; kernel fusion alpha .25,.5,.75 x C .1,1,10',
      test_status='not_run',historical_test_exposed=True,uses_final_wavlm=True)
    write_json(out/'plan.json',plan);audit.update(plan_sha256=sha256(out/'plan.json'),plan_locked_at=plan['locked_at'])
    write_json(out/'input_audit.json',audit)
    with (out/'input_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['sample_id','actor_id','split','label']);w.writerows(zip(raw['sample_ids'],raw['actors'],raw['splits'],raw['labels']))
    code=out/'code';code.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,code/p.name)
    write_json(out/'code_hashes.json',{p.name:sha256(p) for p in code.iterdir()})
    rows=references(out,raw,audit)
    for seed in (42,43,44):
        jobs=[('base',m,'hc','learned') for m in BASE]+[('structure',m,'hc','learned') for m in STRUCTURES]+[('literature',m,'hc','learned') for m in LITERATURE]
        jobs += [('base',m,v,'learned') for v in ('deep','combined') for m in ('visual_only','concat','gated','cross_attention')]
        jobs += [('base','audio_only','hc',a) for a in ('handcrafted','combined')]
        for family,method,v,a in jobs:
            unscaled=view(raw,v,a);stats=fit_normalizers(unscaled,train);d=normalize(unscaled,stats)
            rows.append(fit(out,raw,d,stats,audit,family,method,v,a,seed,source['state_dict']));write_json(out/'runs.json',rows)
    rows+=svms(out,raw,dn,stats0,audit);write_json(out/'runs.json',rows)
    proxy=[dict(method='gated',seed=s,folder=str(out/'gate_sources'/str(s)),val=next(r['val'] for r in rows if r['family']=='base' and r['method']=='gated' and r['visual']=='hc' and r['seed']==s)) for s in (42,43,44)]
    gate_diagnostic(dn,raw,proxy,out)
    means={m:float(np.mean([r['val']['macro_f1'] for r in rows if r['family']=='base' and r['method']==m and r['visual']=='hc' and r['audio']=='learned'])) for m in plan['primary_fusions']}
    selected=max(means,key=means.get)
    lock=dict(locked_at=now(),primary_method=selected,primary_validation_means=means,plan_sha256=audit['plan_sha256'],
      checkpoint_hashes={r['run_id']:sha256(Path(r['folder'])/('checkpoint.joblib' if r['family']=='svm' else 'checkpoint.pt')) for r in rows if r['family']!='reference'},
      decision_grid_sha256=sha256(out/'decision_validation_grid.json'),input_hashes=reader.hashes,test_status='not_run')
    write_json(out/'evaluation_lock.json',lock)
    print('Validation complete; selected',selected,'; hashes locked before loading test numerics.',flush=True)
    test,test_audit=reader.load(['test']);write_json(out/'test_input_audit.json',test_audit)
    with (out/'test_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['sample_id','actor_id','split','label']);w.writerows(zip(test['sample_ids'],test['actors'],test['splits'],test['labels']))
    ix=np.arange(len(test['labels']));y=test['labels']
    for r in rows:
        folder=Path(r['folder']);family=r['family']
        if family=='reference':p=r['alpha']*test['audio_probs']+(1-r['alpha'])*test[f'visual_{r["visual"]}_probs'];pred=None;scores=None
        elif family=='svm':
            ck=joblib.load(folder/'checkpoint.joblib');cfg=ck['config'];dt=normalize(view(test),ck['normalizers']);v=cfg['modality']
            if cfg['kernel']=='precomputed':
                w=cfg['alpha'];xt=w*rbf_kernel(dt['audio_pooled'],dn['audio_pooled'][train],gamma=1/128)+(1-w)*rbf_kernel(dt['visual_pooled'],dn['visual_pooled'][train],gamma=1/320)
            else:xt=dt['audio_pooled'] if v=='audio' else dt['visual_pooled'] if v=='visual' else np.concatenate((dt['audio_pooled'],dt['visual_pooled']),1)
            pred=ck['model'].predict(xt);scores=ck['model'].decision_function(xt);p=None
        else:
            ck=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu');dt=normalize(view(test,r['visual'],r['audio']),ck['normalizers'])
            model=FinalModel(ck['family'],ck['method'],ck['config'],ck['normalizers']);model.load_state_dict(ck['state_dict']);p=infer(model,dt,ix,'cpu');pred=None;scores=None
        r['test']=metrics(y,p if p is not None else np.eye(8)[pred]);r['test_status']='completed'
        save_predictions(folder,test,ix,r['run_id'],p,'test',pred,scores);write_json(folder/'metrics.json',r)
        rec=json.loads((folder/'run.json').read_text());rec['metrics_by_split']['test']=r['test'];rec['test_status']='completed';rec['counts']['actual']['test']=240
        rec['selection']['test_first_evaluated_at']=now();rec['artifacts']['predictions_test_ref']=str((folder/'predictions_test.csv').relative_to(ROOT));rec['test_manifest_ref']=str((out/'test_manifest.csv').relative_to(ROOT));write_json(folder/'run.json',rec)
        if family!='reference':assert sha256(folder/('checkpoint.joblib' if family=='svm' else 'checkpoint.pt'))==lock['checkpoint_hashes'][r['run_id']]
    write_json(out/'runs.json',rows);write_json(out/'status.json',dict(status='completed',records=len(rows),neural_fits=93,svm_fits=27,reference_pipelines=10,
        primary_validation_selected_method=selected,test_status='completed',source_audio_head_reproduced=True,seconds=time.monotonic()-start))
    print('Finished final replay:',len(rows),'records; no test reselection.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('audio','visual','actors','handcrafted','source-checkpoint','development','output'):p.add_argument('--'+arg,required=True)
    run(**vars(p.parse_args()))
