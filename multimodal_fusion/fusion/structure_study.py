"""Predeclared old-feature structure study. Reads development cache only; no test entry point."""
import argparse
import copy
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .paths import project_revision
import numpy as np
import torch
from . import CLASSES
from .data import fit_normalizers, normalize
from .experiment import seed_all, infer, tensor_batch, write_json
from .feature_inputs import sha256
from .feature_pilot import val_metrics
from .models import FusionModel
from .structure_models import StructureModel, METHODS

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def canonical_metrics(m):
    return dict(n=m['n'], accuracy=m['accuracy'], macro_f1=m['macro_f1'],
                macro_precision=m['macro_precision'], macro_recall=m['macro_recall'],
                per_class=[dict(label=i, class_name=c, precision=m['per_class'][c]['precision'],
                                recall=m['per_class'][c]['recall'], f1=m['per_class'][c]['f1-score'],
                                support=int(m['per_class'][c]['support'])) for i,c in enumerate(CLASSES)],
                confusion_matrix_counts=m['confusion_matrix'])


def predictions(path, raw, indices, p, run_id):
    with path.open('w', newline='') as f:
        w=csv.writer(f, lineterminator='\n')
        w.writerow(['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred',
                    *['p_'+c for c in CLASSES]])
        for i, prob in zip(indices, p):
            w.writerow([run_id,'ravdess_speech_v1','val',raw['sample_ids'][i],int(raw['actors'][i]),
                        int(raw['labels'][i]),int(prob.argmax()),*map(float,prob)])


def diagnostics(model, d, val, audio_prediction, visual_prediction):
    probs=infer(model,d,val,'cpu');y=d['labels'][val];pred=probs.argmax(1)
    audio_correct=audio_prediction==y
    result={'per_actor':{str(a):val_metrics(y[d['actors'][val]==a],probs[d['actors'][val]==a])
                         for a in sorted(set(d['actors'][val].tolist()))},
            'repairs_audio':int((~audio_correct & (pred==y)).sum()),
            'harms_audio':int((audio_correct & (pred!=y)).sum()),
            'disagrees_audio':int((pred!=audio_prediction).sum()),
            'either_unimodal_correct_count':int(((audio_prediction==y)|(visual_prediction==y)).sum())}
    for modality in ('audio','visual'):
        # Copy only validation numeric rows: diagnostics never perturb training data.
        small={k:v[val].copy() for k,v in d.items()}
        small[modality+'_pooled'][:]=0;small[modality+'_sequence'][:]=0
        q=infer(model,small,np.arange(len(val)),'cpu')
        result['zero_'+modality]=val_metrics(y,q)
    return result


def fit_one(d, raw, stats, audit, seed, method, base, folder, config, epochs, patience,
            audio_prediction, visual_prediction, plan_hash):
    folder.mkdir(parents=True, exist_ok=False);seed_all(seed);start=now();clock=time.monotonic()
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    model=StructureModel(method,**config)
    if hasattr(model,'base'):
        model.base.load_state_dict(base['state_dict'])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-4)
    rng=np.random.RandomState(seed)  # Matches historical np.random.seed/permutation batch order.
    best=-1.;stale=0;history=[]
    for epoch in range(1,epochs+1):
        model.train();order=rng.permutation(train);loss_sum=0.;main_sum=0.;aux_sum=0.
        for j in range(0,len(order),64):
            ix=order[j:j+64];optimizer.zero_grad()
            logits,aux=model(tensor_batch(d,ix,'cpu'),return_aux=True)
            y=torch.as_tensor(d['labels'][ix],dtype=torch.long)
            main=torch.nn.functional.cross_entropy(logits,y)
            extra=sum((torch.nn.functional.cross_entropy(z,y) for z in aux),torch.zeros(()))
            loss=main+config['aux_weight']*extra
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            loss_sum+=float(loss.detach())*len(ix);main_sum+=float(main.detach())*len(ix);aux_sum+=float(extra.detach())*len(ix)
        m=val_metrics(d['labels'][val],infer(model,d,val,'cpu'))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),train_main_ce=main_sum/len(train),
                            train_aux_ce=aux_sum/len(train),val_macro_f1=m['macro_f1'],val_accuracy=m['accuracy']))
        if m['macro_f1']>best+1e-8:
            best=m['macro_f1'];best_epoch=epoch;state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(state)
    if hasattr(model,'base'):
        for k,v in model.base.state_dict().items():torch.testing.assert_close(v,base['state_dict'][k],rtol=0,atol=0)
    p=infer(model,d,val,'cpu');m=val_metrics(d['labels'][val],p)
    rid=f'M5_structure_{method}_s{seed}_20261020'
    checkpoint=dict(method=method,config=config,state_dict=state,normalizers=stats,seed=seed,best_epoch=best_epoch,
                    class_names=list(CLASSES),source_hashes=audit['source_hashes'],test_status='not_run',plan_sha256=plan_hash)
    torch.save(checkpoint,folder/'checkpoint.pt')
    ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
    restored=StructureModel(ck['method'],**ck['config']);restored.load_state_dict(ck['state_dict'])
    np.testing.assert_allclose(p,infer(restored,d,val,'cpu'),rtol=1e-5,atol=1e-6)
    predictions(folder/'predictions_val.csv',raw,val,p,rid)
    diag=diagnostics(model,d,val,audio_prediction,visual_prediction)
    write_json(folder/'diagnostics_val.json',diag);write_json(folder/'history.json',history)
    active=sum(p.numel() for p in model.parameters() if p.requires_grad)
    unused=0
    if method=='uniform_attention':
        h=config['hidden'];unused=2*h*h+2*h  # Q/K receive no gradients under uniform weights.
    if method.startswith('audio_residual') and 'aux' not in method:
        unused=sum(p.numel() for module in (model.audio_aux,model.visual_aux) for p in module.parameters())
    row=dict(method=method,seed=seed,val=canonical_metrics(m),test=None,test_status='not_run',
             best_epoch=best_epoch,completed_epochs=len(history),parameters_total=sum(p.numel() for p in model.parameters()),
             parameters_requires_grad=active,parameters_effectively_used=active-unused,
             seconds=time.monotonic()-clock,checkpoint_roundtrip_verified=True,repairs_audio=diag['repairs_audio'],
             harms_audio=diag['harms_audio'],folder=str(folder),run_id=rid)
    if hasattr(model,'scale'):row['residual_scale']=float(model.scale.detach())
    write_json(folder/'metrics.json',row)
    record=json.loads((ROOT/'docs/templates/run.template.json').read_text())
    record.update(run_id=rid,member='M5',owner='Jiang Yushan',stage='fusion',comparison_group='old_features_structure_exploration_20261020',
                  pipeline_name=method,run_status='completed',test_status='not_run',started_at=start,finished_at=now(),seed=seed,
                  split_sha256=audit['source_hashes']['actor_csv'],input_manifest_ref=str(folder.parent.parent/'input_manifest.csv'),
                  input_manifest_sha256=sha256(folder.parent.parent/'input_manifest.csv'),
                  git_commit=project_revision(ROOT),working_tree_dirty=True,
                  code_patch_ref=str(folder.parent.parent/'code'),command=' '.join(sys.argv))
    record['counts']['actual']={'train':len(train),'val':len(val),'test':None}
    record['environment'].update(python=platform.python_version(),framework_and_version=str(torch.__version__),device='cpu')
    record['source_runs']['audio_learned']='M2_encoder_transformer_s42_20261015_00'
    record['input_artifacts']=[dict(source_hashes=audit['source_hashes'],development_cache_sha256=audit['development_cache_sha256'])]
    record['preprocessing'].update(standardization_fit_split='train',augmentation_train_only=True,
                                   parameters=dict(modality_dropout=.25 if 'mdrop' in method else 0))
    extractor=record['pipeline']['feature_extractors'][0]
    extractor.update(component_id='historical_audio_and_visual',modality='audio+visual',kind='cached',
                     method='M1 48D + MFCC Transformer128D; MobileNet576D',trainable_in_this_run=False,
                     input_description='Existing paired development cache; no WavLM, probabilities or test rows')
    residual=method.startswith('audio_residual')
    record['pipeline']['representation'].update(kind='combined',pooling='supplied pooled' if residual else 'masked sequence',
                    normalization='per-dimension standardization',normalization_fit_split='train',
                    components=[dict(modality='audio',dim=config['audio_dim'] if residual else config['audio_seq_dim']),
                                dict(modality='visual',dim=config['visual_dim'] if residual else config['visual_seq_dim'])])
    record['pipeline']['fusion'].update(applied=True,method=method,parameters=config,trainable_in_this_run=True)
    record['pipeline']['classifier'].update(type='frozen_audio_MLP_plus_residual' if residual else 'MLP',
                     hidden_dims=[config['hidden']],activation='ReLU',dropout=config['dropout'],trainable_in_this_run=True,probability_method='softmax')
    record['training'].update(regime='frozen_features_and_audio_head' if residual else 'frozen_features',optimizer='AdamW',
                    learning_rate=.001,batch_size=64,max_epochs=epochs,completed_epochs=len(history),early_stopping_patience=patience,
                    class_weighting='none',other_parameters=dict(weight_decay=1e-4,gradient_clip=1.,
                    auxiliary_ce_weight_each=.2 if 'aux' in method else 0,plan_sha256=plan_hash,batch_order_rng='numpy.RandomState(seed), same historical batch order'))
    record['selection'].update(best_epoch=best_epoch,tie_break_rule='retain earlier epoch',config_locked_at=audit['plan_locked_at'],
                               selection_reason='Validation macro-F1; exploratory after historical test exposure')
    record['metrics_by_split']={'val':row['val'],'test':None}
    record['artifacts'].update(predictions_val_ref=str(folder/'predictions_val.csv'),predictions_test_ref=None,
                    checkpoint_ref=str(folder/'checkpoint.pt'),checkpoint_sha256=sha256(folder/'checkpoint.pt'),
                    history_ref=str(folder/'history.json'),original_metrics_ref=str(folder/'metrics.json'))
    record['limitations']=['Historical test already exposed; this run evaluates validation only.',
           'Residual variants use audio176D; sequence variants use learned audio128D; compare within each family.',
           'Relative positions use valid-token index, not physical timestamps; upstream trimming provenance remains unknown.',
           'Visual tokens treated valid as in original cache; no new visual QA claimed.',
           'Same allocated parameter count is not the same effective capacity; uniform Q/K and unused auxiliary heads disclosed.']
    record['frozen_audio_reference']=audit['base_references'][str(seed)] if residual else None
    (folder/'notes.md').write_text(f'# {rid}\n\nValidation-only exploratory structure comparison; no WavLM or test inference.\n\n'
              f'Best epoch {best_epoch}; frozen feature sources and normalizers recorded in checkpoint/run.json.\n'
              f'Parameters total/gradient/effective: {row["parameters_total"]}/{active}/{active-unused}.\n'
              'See study plan.json for predeclared comparisons and limitations.\n')
    write_json(folder/'run.json',record)
    print(f'{method} seed{seed}: valF1={best:.4f} acc={m["accuracy"]:.4f} epoch={best_epoch}/{len(history)}',flush=True)
    return row


def run(pilot, base_study, output, epochs=30, patience=8, seeds=(42,43,44), methods=METHODS):
    seed_all(42)
    pilot=Path(pilot).resolve();base_study=Path(base_study).resolve();out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    with np.load(pilot/'development_features.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    if len(raw['labels'])!=1200 or set(raw['splits'].tolist())!={'train','val'} or any('prob' in k for k in raw):
        raise ValueError('Expected only 960 train/240 val cached feature rows, no probabilities')
    audit=json.loads((pilot/'adapter_audit.json').read_text());audit['development_cache_sha256']=sha256(pilot/'development_features.npz')
    assert sha256(ROOT/'multimodal_fusion/configs/actor_splits.csv')==audit['source_hashes']['actor_csv']
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    assert len(train)==960 and len(val)==240 and not (set(raw['actors'][train]) & set(raw['actors'][val]))
    stats=fit_normalizers(raw,train);d=normalize(raw,stats)
    config=dict(audio_dim=raw['audio_pooled'].shape[-1],visual_dim=raw['visual_pooled'].shape[-1],
                audio_seq_dim=raw['audio_sequence'].shape[-1],visual_seq_dim=raw['visual_sequence'].shape[-1],
                hidden=64,heads=4,dropout=.2,modality_dropout=.25,aux_weight=.2)
    old=json.loads((base_study/'default_runs.json').read_text())
    baselines=[];references={};loaded={}
    for seed in seeds:
        loaded[seed]={}
        for method in ('audio_only','visual_only','concat','gated','cross_attention','sequence_mean_concat'):
            prior=next(r for r in old if r['method']==method and r['seed']==seed)
            path=Path(prior['folder'])/'checkpoint.pt'
            if not path.is_absolute():path=ROOT/'multimodal_fusion'/path
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['normalizers']==stats and ck['source_hashes']==audit['source_hashes']
            m=FusionModel(method,**ck['config']);m.load_state_dict(ck['state_dict'])
            p=infer(m,d,val,'cpu');score=val_metrics(d['labels'][val],p)
            assert abs(score['macro_f1']-prior['val']['macro_f1'])<1e-12
            loaded[seed][method]=(ck,p)
            baselines.append(dict(method=method,seed=seed,val=canonical_metrics(score),checkpoint_sha256=sha256(path),reused=True))
            if method=='audio_only':references[str(seed)]={'checkpoint_ref':str(path),'checkpoint_sha256':sha256(path),'best_epoch':ck['best_epoch']}
    locked=now();audit.update(base_references=references,plan_locked_at=locked)
    plan=dict(locked_at=locked,methods=list(methods),seeds=list(seeds),configuration=config,
              max_epochs=epochs,patience=patience,learning_rate=.001,weight_decay=1e-4,batch_size=64,
              validation_only=True,test_status='not_run',historical_test_already_exposed=True,uses_wavlm=False,
              comparisons=['residual 2x2 factorial: visual dropout(.25) x auxiliary CE(.2 each)',
                 'sequence controls: projected mean, uniform attention retraining, relative position, depthwise temporal convolution, bidirectional attention'],
              selection='No adaptive candidates or test evaluation. Report all seeds; >=.02 mean val gain and >=2/3 positive seeds is a practical followup gate, not significance.',
              residual_reference='Frozen seed-matched historical audio176D MLP, validation-selected checkpoint; original dropout disabled.',
              position_caveat='Relative valid-token positions, not recovered real timestamps. No precise alignment claim.',
              future_replay='New feature export/cache + matching newly trained audio-only checkpoints required; never mix old/new source tensors.',
              input_cache_sha256=audit['development_cache_sha256'],source_hashes=audit['source_hashes'],base_references=references)
    write_json(out/'plan.json',plan);plan_hash=sha256(out/'plan.json')
    write_json(out/'source_audit.json',audit);write_json(out/'reused_baselines.json',baselines)
    shutil.copyfile(pilot/'input_manifest.csv',out/'input_manifest.csv')
    code=out/'code';code.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,code/p.name)
    write_json(out/'code_hashes.json',{p.name:sha256(p) for p in code.iterdir()})
    rows=[]
    for seed in seeds:
        for method in methods:
            row=fit_one(d,raw,stats,audit,seed,method,loaded[seed]['audio_only'][0],out/'runs'/f'{method}_s{seed}',
                        config,epochs,patience,loaded[seed]['audio_only'][1].argmax(1),loaded[seed]['visual_only'][1].argmax(1),plan_hash)
            rows.append(row);write_json(out/'runs.json',rows)
    report(rows,baselines,out)
    write_json(out/'status.json',dict(run_status='completed',new_fits=len(rows),reused_baselines=len(baselines),
               test_status='not_run',uses_wavlm=False,seconds=time.monotonic()-start,plan_sha256=plan_hash))


def report(rows, baselines, out):
    summary={}
    for method in [*dict.fromkeys(r['method'] for r in baselines+rows)]:
        runs=[r for r in baselines+rows if r['method']==method]
        f1=np.array([r['val']['macro_f1'] for r in runs]);acc=np.array([r['val']['accuracy'] for r in runs])
        baseline='audio_only' if method.startswith('audio_residual') else 'cross_attention'
        diff=[r['val']['macro_f1']-next(b['val']['macro_f1'] for b in baselines if b['seed']==r['seed'] and b['method']==baseline) for r in runs]
        summary[method]=dict(n=len(runs),seeds=[r['seed'] for r in runs],macro_f1_mean=float(f1.mean()),
                 macro_f1_std=float(f1.std(ddof=1)) if len(f1)>1 else None,accuracy_mean=float(acc.mean()),
                 comparison_reference=baseline,paired_f1_differences=diff,
                 followup_gate_met=bool(np.mean(diff)>=.02 and sum(x>0 for x in diff)>=2))
    write_json(out/'summary.json',summary)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['method','n','val_macro_f1_mean','val_macro_f1_sd','val_accuracy_mean','test_status'])
        for method,s in summary.items():w.writerow([method,s['n'],s['macro_f1_mean'],s['macro_f1_std'],s['accuracy_mean'],'not_run'])
    text=['# 旧特征结构改进：验证集实验\n','本轮未接WavLM、未评价测试集。既有测试已暴露，因此本轮属于探索。\n',
          '| 方法 | 验证 Macro-F1（均值±样本SD） | 验证 Accuracy |','|---|---:|---:|']
    for method,s in summary.items():
        sd=s['macro_f1_std']
        text.append(f'| {method} | {s["macro_f1_mean"]:.4f} ± {sd:.4f} | {s["accuracy_mean"]:.4f} |' if sd is not None else f'| {method} | {s["macro_f1_mean"]:.4f} | {s["accuracy_mean"]:.4f} |')
    text+=['\n残差组使用176D音频pooled，序列组使用128D learned音频，不能把跨组差异全部归因于融合结构。',
       '残差组冻结seed对应旧音频MLP，独立对照visual dropout和辅助损失；辅助视觉头使用未丢弃的视觉嵌入。',
       'Uniform保留同样分配参数，但Q/K无梯度；实际有效参数量记录在每个run。Uniform不施加attention权重dropout。',
       '位置编码仅为相对token位置，不是真实时间戳。双向方法参数更多，不能声称机制比较已完全控制容量。',
       '所有baseline均重算验证预测并与历史分数核对，无重训。27个新run的checkpoint/normalizer/代码快照/逐样本预测独立保存。',
       '后续输入定稿后必须重建开发cache、重训对应audio-only基线，再重跑该流程，保留本次历史结果。']
    (out/'实验结果.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot',required=True);p.add_argument('--base-study',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=8)
    a=p.parse_args();run(**vars(a))
