"""Publish existing M5 results in the team's schema v2. Does not train/evaluate models."""
import argparse
import copy
import csv
import difflib
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASSES=['neutral','calm','happy','sad','angry','fearful','disgust','surprise']
OLD_M2='M2_encoder_transformer_s42_20261015_00'
BASE_COMMIT='6ea5fa37bd0f56847ba5755dcab2671845d4dcc6'
GROUP_FUSION='M5_fusion_mobilenet_audio176_v1'
GROUP_CONTROL='M5_pooled_controls_v1'
GROUP_SEQUENCE='M5_sequence_controls_v1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def ref(path,dest):return Path(path).relative_to(dest).as_posix()
def safe_text(s):
    s=re.sub(r'/Users/jiang/[^\s\n\"\)\>]*','[local artifact omitted]',s)
    return s


def standardized_predictions(source,target,run_id):
    with Path(source).open() as f:rows=list(csv.DictReader(f))
    fields=['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred',*['p_'+c for c in CLASSES]]
    converted=[]
    for row in rows:
        sid=row['sample_id'];lab=int(row['label']);pred=int(row.get('prediction',row.get('y_pred',-1)))
        converted.append(dict(zip(fields,[run_id,'ravdess_speech_v1',row.get('split','val'),sid,
                int(row.get('actor_id',sid[-2:])),lab,pred,*[float(row[c]) for c in CLASSES]])))
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(converted)
    return converted


def measure(rows):
    y=np.asarray([r['y_true'] for r in rows]);pred=np.asarray([r['y_pred'] for r in rows])
    p=np.asarray([[r['p_'+c] for c in CLASSES] for r in rows])
    assert len({(r['run_id'],r['split'],r['sample_id']) for r in rows})==len(rows)
    assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-5)
    assert np.array_equal(p.argmax(1),pred)
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
    return {'n':len(y),'accuracy':float(accuracy_score(y,pred)),'macro_precision':float(precision.mean()),
       'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
       'per_class':[{'label':i,'class_name':c,'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(CLASSES)],
       'confusion_matrix_counts':confusion_matrix(y,pred,labels=range(8)).tolist()}


def extractor(component,modality,kind,method,pooled,sequence=None,source=None):
    return {'component_id':component,'modality':modality,'kind':kind,'method':method,'input_description':'Frozen teammate export; original extractor training not part of this run',
       'parameters':{},'pretrained_source':None,'source_run_id':source,'checkpoint_ref':None,'checkpoint_sha256':None,
       'trainable_in_this_run':False,'trainable_layers':[],'output_sequence_shape':sequence,'output_pooled_shape':pooled}


def export(source,dest,upstream):
    source=Path(source).resolve();dest=Path(dest).resolve();upstream=Path(upstream).resolve()
    project=source/'multimodal_fusion';records=dest/'experiment_records';shared=records/'shared/M5';shared.mkdir(parents=True,exist_ok=True)
    template=read(upstream/'docs/templates/run.template.json')
    study=project/'results/m5_validation_study_20261019';final=project/'results/m5_final_evaluation_20261019';pilot=project/'results/m5_feature_pilot_20261019_s42'
    defaults=read(study/'default_runs.json');tuning=read(study/'tuning_runs.json');final_result=read(final/'results.json')
    historical_env=read(pilot/'validation_metrics.json')['environment']
    confirmation=read(final/'class_order_confirmation.json');assert confirmation['input_identity_verified']
    audit=read(study/'source_audit.json');split_ref='multimodal_fusion/configs/actor_splits.csv'
    # Full input manifest is metadata, not raw features or model weights.
    manifests=[]
    for path in [pilot/'input_manifest.csv',final/'test_manifest.csv']:
        with path.open() as f:manifests.extend(csv.DictReader(f))
    with (shared/'input_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','actor_id','split','label'],lineterminator='\n');w.writeheader();w.writerows(sorted(manifests,key=lambda x:x['sample_id']))
    assert len(manifests)==1440 and len({r['sample_id'] for r in manifests})==1440
    manifest_ref=ref(shared/'input_manifest.csv',dest);manifest_hash=digest(shared/'input_manifest.csv')
    with (shared/'environment.txt').open('w') as f:
        f.write('Publication validation runtime (historical per-run fields retained separately)\n')
        f.write(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True))
    # Strip host-specific direct-reference lines from environment, not dependency versions.
    p=shared/'environment.txt';p.write_text('\n'.join(l for l in p.read_text().splitlines() if '@ file:' not in l and '/Users/' not in l)+'\n')
    # Source snapshots are deduplicated by content, retaining training-time versions where available.
    snapshots={}
    for folder in [pilot/'code',study/'code',study/'diagnostic_code',final/'code']:
        for p in folder.glob('*.py'):
            dst=shared/'source_snapshots'/digest(p)/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
            snapshots.setdefault(folder.as_posix(),[]).append(ref(dst,dest))
    write(shared/'source_snapshot_index.json',{Path(k).relative_to(source).as_posix():v for k,v in snapshots.items()})
    patch=[]
    for p in sorted((project/'fusion').glob('*.py'))+sorted((project/'tests').glob('test_*.py')):
        relative=p.relative_to(source).as_posix()
        old=subprocess.run(['git','show',BASE_COMMIT+':'+relative],cwd=source,capture_output=True,text=True)
        original=old.stdout if old.returncode==0 else ''
        patch.extend(difflib.unified_diff(original.splitlines(keepends=True),p.read_text().splitlines(keepends=True),
                       fromfile='a/'+relative if old.returncode==0 else '/dev/null',tofile='b/'+relative))
    (shared/'historical_code.patch').write_text(''.join(patch))
    # Small protocol/diagnostic artifacts only. Paths inside copied JSON are sanitized below.
    for name,path in [('validation_plan.json',study/'plan.json'),('tuning_decisions.json',study/'tuning_decisions.json'),
                      ('decision_validation_grid.json',final/'decision_validation_grid.json'),('class_order_confirmation.json',final/'class_order_confirmation.json')]:
        obj=read(path)
        write(shared/name,obj)
    write(shared/'evaluation_lock.json',{
        'locked_at':read(final/'locked_plan.json')['locked_at'],
        'primary_method_selected_by_validation':'cross_attention',
        'strongest_matched_unimodal_selected_by_validation':'visual_only',
        'candidates':[{'method':x['method'],'seed':x['seed'],'best_epoch':x['best_epoch'],'checkpoint_sha256':x['checkpoint_sha256']} for x in read(final/'locked_plan.json')['candidates']],
        'decision_alpha_audio':.15,'decision_weight_selection_split':'val','no_test_based_reselection':True,
        'class_order_confirmation_after_evaluation':True})
    for p in (study/'diagnostics').glob('*.json'):write(shared/'diagnostics'/p.name,read(p))
    gate_text=(study/'diagnostics/gates_val.csv').read_text()
    (shared/'gates_val.csv').write_text(gate_text)
    for folder,label in [(study/'figures','validation'),(final/'figures','final')]:
        target=shared/'figures'/label;target.mkdir(parents=True,exist_ok=True)
        for p in folder.glob('*.png'):shutil.copyfile(p,target/p.name)
    # Store artifact hashes without pretending local weights/features are retrievable from GitHub.
    assets=[]
    def asset(role,path):
        path=Path(path);entry={'role':role,'filename':path.name,'sha256':digest(path),
                             'availability':'Local M5 artifact; not uploaded. Request from M5 owner.', 'download_ref':None}
        assets.append(entry);return entry
    asset('frozen_visual_features',source/'features/mobilenetv3_small_features.npz')
    asset('combined_audio_features',source/'features/Audio_Handoff/features/audio_fused_176d.csv')
    source_artifacts=[{'role':'audio_combined_export','source_run_id':None,'ref':None,'sha256':audit['source_hashes']['audio_csv']},
                     {'role':'visual_learned_export','source_run_id':None,'ref':None,'sha256':audit['source_hashes']['visual_npz']},
                     {'role':'ordered_train_val_audio_sequences','source_run_id':OLD_M2,'ref':None,'sha256':audit['source_hashes']['ordered_train_val_audio_sequences']}]
    all_records=[];source_map={};export_baseline={}
    def base(run_id,group,seed):
        r=copy.deepcopy(template);r.update(run_id=run_id,member='M5',owner='Jiang Yushan',stage='fusion',comparison_group=group,
            run_status='completed',test_status='not_run',seed=seed,git_commit=BASE_COMMIT,working_tree_dirty=True,
            code_patch_ref=ref(shared/'historical_code.patch',dest),split_ref=split_ref,split_sha256=audit['source_hashes']['actor_csv'],
            input_manifest_ref=manifest_ref,input_manifest_sha256=manifest_hash,command=None)
        r['environment']={'python':None,'framework_and_version':None,'device':'cpu','dependencies_ref':ref(shared/'environment.txt',dest)}
        r['excluded_samples']={'count':0,'id_reason_manifest_ref':None,'handling_policy':'No samples excluded. Unmeasured test results remain null.'}
        r['counts']['actual']={'train':960,'val':240,'test':None}
        r['notes_ref']=f'experiment_records/{run_id}/notes.md'
        r['input_artifacts']=copy.deepcopy(source_artifacts)
        r['source_runs']['audio_learned']=OLD_M2
        return r
    def prediction_pair(r,folder,sources,expected):
        for split,source_csv in sources.items():
            converted=standardized_predictions(source_csv,folder/f'predictions_{split}.csv',r['run_id'])
            m=measure(converted);orig=expected[split]
            assert abs(m['macro_f1']-orig['macro_f1'])<1e-12
            assert m['confusion_matrix_counts']==orig['confusion_matrix']
            r['metrics_by_split'][split]=m;r['artifacts'][f'predictions_{split}_ref']=ref(folder/f'predictions_{split}.csv',dest)
        if 'test' not in sources:r['metrics_by_split']['test']=None
        else:r['test_status']='completed';r['counts']['actual']['test']=240
    def save(r,folder,description,limitation):
        folder.mkdir(parents=True,exist_ok=True);write(folder/'run.json',r)
        val=r['metrics_by_split']['val'];test=r['metrics_by_split']['test'];head=r['pipeline']['classifier']
        cm=np.array((test or val)['confusion_matrix_counts']);pairs=sorted([(int(cm[i,j]),CLASSES[i],CLASSES[j]) for i in range(8) for j in range(8) if i!=j],reverse=True)[:2]
        text=f'''# {r['run_id']} — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

{description}

Comparison group: `{r['comparison_group']}`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | {r['pipeline_name']} | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | {r['pipeline']['representation']['kind']} / {r['pipeline']['representation']['output_shape']} | Train-only normalization where applicable |
| Fusion | {r['pipeline']['fusion']['method'] or 'none'} | {r['pipeline']['fusion']['trainable_in_this_run']} |
| Classifier | {head['architecture'] or head['type'] or 'Upstream architecture not supplied'} | {r['training']['regime']} |

## 3. Results and selection

Validation Macro-F1={val['macro_f1']:.4f}; test {'Macro-F1='+format(test['macro_f1'],'.4f') if test else 'not_run'}. Best epoch={r['selection']['best_epoch']}. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

{limitation} The largest {'test' if test else 'validation'} confusion counts include {pairs[0][1]}→{pairs[0][2]} ({pairs[0][0]}) and {pairs[1][1]}→{pairs[1][2]} ({pairs[1][0]}). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
'''
        (folder/'notes.md').write_text(text);all_records.append(r)
    # 25 distinct neural configuration/seed results (21 default, 4 validation-only tuning).
    for row in defaults+tuning:
        origin=project/row['folder'];origin=origin if origin.exists() else Path(row['folder'])
        old=read(origin/'run.json');rid=old['run_id'];source_map[(row['method'],row['seed'],row['hidden'],row['dropout'])]=rid
        method=row['method'];is_tuning=row in tuning
        group='M5_regularization_exploration_v1' if is_tuning else GROUP_FUSION if method in ('concat','gated','cross_attention') else GROUP_CONTROL if method in ('audio_only','visual_only') else GROUP_SEQUENCE
        r=base(rid,group,row['seed']);folder=records/rid;folder.mkdir(parents=True,exist_ok=True)
        ck=torch.load(origin/'checkpoint.pt',weights_only=True,map_location='cpu');config=ck['config'];hidden=config['hidden'];dropout=config['dropout']
        r['pipeline_name']=method+' on frozen historical audio/visual exports'
        uses_audio=method!='visual_only';uses_visual=method not in ('audio_only','audio_sequence_mean')
        uses_seq=method in ('cross_attention','audio_sequence_mean','sequence_mean_concat')
        r['source_runs']['audio_learned']=OLD_M2 if uses_audio else None
        roles=({'ordered_train_val_audio_sequences'} if uses_seq else {'audio_combined_export'}) if uses_audio else set()
        if uses_visual:roles.add('visual_learned_export')
        r['input_artifacts']=[x for x in r['input_artifacts'] if x['role'] in roles]
        components=[]
        if uses_audio:
            if not uses_seq:components.append(extractor('audio_handcrafted','audio','handcrafted','M1 time/spectral/DWT descriptors',[None,48]))
            components.append(extractor('audio_learned','audio','learned','MFCC Transformer',[None,128],[None,None,128] if uses_seq else None,OLD_M2))
        if uses_visual:components.append(extractor('visual_learned','visual','learned','MobileNetV3-Small per supplied export',[None,576],[None,16,576] if uses_seq else None))
        fusion=method in ('concat','gated','cross_attention','sequence_mean_concat')
        input_dim={'audio_only':176,'visual_only':576,'concat':752,'gated':hidden,'cross_attention':hidden,'audio_sequence_mean':128,'sequence_mean_concat':704}[method]
        kind='combined' if method in ('concat','gated','cross_attention','sequence_mean_concat','audio_only') else 'learned'
        r['pipeline']['feature_extractors']=components
        r['pipeline']['representation'].update(kind=kind,components=[c['component_id'] for c in components],
            pooling='masked temporal mean' if uses_seq else 'supplied temporal mean pooled features',
            normalization='per-channel standardization (sequence statistics exclude padding)',normalization_fit_split='train',
            combination_method=method,output_shape=[None,input_dim])
        r['pipeline']['fusion'].update(applied=fusion,method=method if fusion else None,
            input_source_runs={'audio':None,'visual':None} if fusion else {},
            parameters={'audio_pooled_dim':176,'visual_pooled_dim':576,'audio_sequence_dim':128,'visual_sequence_dim':576,'hidden':hidden,'heads':4 if method=='cross_attention' else None},
            trainable_in_this_run=method in ('gated','cross_attention'),output_shape=[None,input_dim] if fusion else None)
        r['pipeline']['classifier'].update(type='MLP',input_dim=input_dim,architecture=f'Linear({input_dim},{hidden}) → ReLU → Dropout({dropout}) → Linear({hidden},8)',
            hidden_dims=[hidden],output_dim=8,activation='ReLU',dropout=dropout,parameters={'loss':'cross_entropy','label_smoothing':0.0},
            trainable_in_this_run=True,probability_method='softmax; argmax ties choose lowest index',checkpoint_sha256=digest(origin/'checkpoint.pt'))
        asset('M5 checkpoint '+rid,origin/'checkpoint.pt')
        norm_path=shared/'normalizers'/ (digest_json(ck['normalizers'])+'.json');write(norm_path,ck['normalizers'])
        r['preprocessing'].update(audio='Frozen export; no raw preprocessing rerun',visual='Frozen export; 16 tokens assumed valid, pending mask provenance',
            standardization_fit_split='train',augmentation_train_only=False,parameters={'normalizer_ref':ref(norm_path,dest),'normalizer_sha256':digest(norm_path)})
        r['training'].update(regime='frozen_encoder_then_classifier',optimizer='AdamW',learning_rate=.001,batch_size=64,
            max_epochs=30,completed_epochs=row['completed_epochs'],early_stopping_patience=8,class_weighting='none',
            other_parameters={'weight_decay':1e-4,'gradient_clip':1.,'parameter_count':row.get('parameters'),'hidden':hidden,'dropout':dropout})
        r['selection'].update(best_epoch=row['best_epoch'],selected_for_next_stage=(method=='cross_attention' and not is_tuning) if group==GROUP_FUSION else None,
            selection_reason='Method-level three-seed validation mean; test ranking does not change selection' if group==GROUP_FUSION else 'Control / exploratory configuration; no test-based selection',
            tie_break_rule='earlier epoch',candidate_search_ref=ref(shared/'tuning_decisions.json',dest) if is_tuning else None,
            config_locked_at=read(final/'locked_plan.json')['locked_at'] if not is_tuning else None,
            test_first_evaluated_at=final_result['test_started_at'] if not is_tuning else None)
        sources={'val':origin/'predictions_val.csv'};expected={'val':row['val']}
        if not is_tuning:
            fr=next(x for x in final_result['models'] if x['method']==method and x['seed']==row['seed'])
            sources['test']=final/f'{method}_s{row["seed"]}'/'predictions_test.csv';expected['test']=fr['test']
        prediction_pair(r,folder,sources,expected)
        history=read(origin/'history.json');r['artifacts']['history_ref']=ref(folder/'history.csv',dest)
        with (folder/'history.csv').open('w',newline='') as f:
            keys=list(history[0]);w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(history)
        r['artifacts']['checkpoint_sha256']=digest(origin/'checkpoint.pt')
        r['environment'].update(python=historical_env['python'],framework_and_version='PyTorch '+historical_env['torch'])
        original_command=old.get('command') or ''
        module='fusion.feature_pilot' if 'feature_pilot' in original_command else 'fusion.study'
        suffix=original_command[original_command.find('--'):] if '--' in original_command else '(see source snapshot and training settings)'
        r['command']='python -m '+module+' '+suffix
        write(folder/'original_metrics.json',{'val':expected['val'],'test':expected.get('test')});r['artifacts']['original_metrics_ref']=ref(folder/'original_metrics.json',dest)
        limitation='Only validation was evaluated for this exploratory regularization trial.' if is_tuning else 'Test was evaluated only after the candidate/epoch lock; test actors are now exposed for future work.'
        if uses_seq:limitation+=' Sequence inputs and projection/normalization differ from pooled baselines; performance changes do not isolate attention alone.'
        save(r,folder,'Compare '+method+' against fixed-feature controls under the same actor split and training budget.',limitation)
    # Explicitly referenced evaluations of upstream heads, not newly trained M5 models.
    for key,modality in [('upstream_audio','audio'),('upstream_visual_assumed_order','visual')]:
        rid=f'M5_reference_{modality}_head_sNA_20261019_01';export_baseline[modality]=rid
        r=base(rid,'M5_imported_probability_references_v1',None);r['stage']='reference'
        if modality=='visual':r['source_runs']['audio_learned']=None
        r['input_artifacts']=[x for x in r['input_artifacts'] if x['role']==('visual_learned_export' if modality=='visual' else 'audio_combined_export')]
        r['pipeline_name']=f'Imported {modality} feature export → supplied upstream classifier probabilities (no M5 retraining)'
        r['pipeline']['feature_extractors']=[]
        r['pipeline']['representation'].update(kind='probabilities',components=[],pooling=None,normalization=None,normalization_fit_split=None,combination_method=None,output_shape=[None,8])
        r['pipeline']['fusion'].update(applied=False,method=None,trainable_in_this_run=False)
        r['pipeline']['classifier'].update(type='upstream classifier (architecture not supplied)',input_dim=176 if modality=='audio' else 576,output_dim=8,
            architecture=None,trainable_in_this_run=False,probability_method='provided probabilities; class order confirmed')
        r['training']['regime']='cached_upstream_evaluation';r['counts']['actual']['train']=None
        r['selection'].update(selected_for_next_stage=None,selection_reason='Reference evaluation of a supplied upstream head; no new training or model selection')
        folder=records/rid;original=final_result['decision'][key]
        prediction_pair(r,folder,{s:final/key/f'predictions_{s}.csv' for s in ['val','test']},original)
        save(r,folder,'Reference the probability-producing head supplied by the teammate, without claiming a new M5 training run.',
             'Upstream training records/checkpoint identity are not fully supplied. Probability class order has been confirmed, but that does not establish identity with the feature extractor checkpoint.')
    # Two decision configurations; no new classifier/encoder training.
    for key in ('decision_equal','decision_tuned'):
        original=final_result['decision'][key];rid=f'M5_fusion_{key}_sNA_20261019_01';r=base(rid,'M5_decision_fixed_exports_v1',None)
        r['pipeline_name']=key+' using supplied audio/visual heads';r['pipeline']['feature_extractors']=[]
        r['source_runs']['audio_final']=export_baseline['audio'];r['source_runs']['visual_learned']=export_baseline['visual']
        r['pipeline']['representation'].update(kind='probabilities',components=[],pooling=None,normalization=None,normalization_fit_split=None,combination_method='weighted probability mean',output_shape=[None,8])
        r['pipeline']['fusion'].update(applied=True,method=key,input_source_runs={'audio':export_baseline['audio'],'visual':export_baseline['visual']},
            parameters={'alpha_audio':original['alpha_audio'],'alpha_visual':1-original['alpha_audio'],'class_order_confirmation_ref':ref(shared/'class_order_confirmation.json',dest)},trainable_in_this_run=False,output_shape=[None,8])
        r['pipeline']['classifier'].update(type='none',input_dim=None,hidden_dims=None,output_dim=None,activation=None,dropout=None,trainable_in_this_run=False,
            probability_method='weighted average of supplied probabilities')
        r['training']['regime']='probability_fusion';r['counts']['actual']['train']=None
        r['selection'].update(selected_for_next_stage=(key=='decision_tuned'),selection_reason='Weight chosen on validation only' if key=='decision_tuned' else 'Predefined 0.5/0.5 reference',
            tie_break_rule='closest audio weight to 0.5',candidate_search_ref=ref(shared/'decision_validation_grid.json',dest),
            config_locked_at=read(final/'locked_plan.json')['locked_at'],test_first_evaluated_at=final_result['test_started_at'])
        folder=records/rid
        prediction_pair(r,folder,{s:final/key/f'predictions_{s}.csv' for s in ['val','test']},original)
        save(r,folder,'Compare fixed equal weighting and validation-tuned probability weighting, holding source probability exports fixed.',
             'Class order was confirmed after the initial evaluation and values remained unchanged. Weight selection used validation only; test performance did not change alpha. No new classifier was trained.')
    # Real audio pilot: new 128D head is a distinct run; original M2 baseline is referenced, not duplicated.
    pilot_audio=project/'results/m2_audio_pilot_20261016';old=read(pilot_audio/'validation_metrics.json')
    rid='M5_audio_learned_pilot_s42_20261016_01';r=base(rid,'M5_audio_interface_pilot_v1',42);r['stage']='representation'
    r['pipeline_name']='Frozen M2 Transformer128D → newly trained MLP128→64→8'
    r['pipeline']['feature_extractors']=[extractor('audio_learned','audio','learned','MFCC Transformer',[None,128],[None,None,128],OLD_M2)]
    r['pipeline']['representation'].update(kind='learned',components=['audio_learned'],pooling='upstream temporal mean',normalization='train-only per-dimension standardization',normalization_fit_split='train',combination_method=None,output_shape=[None,128])
    r['pipeline']['fusion'].update(applied=False,method=None,trainable_in_this_run=False)
    r['pipeline']['classifier'].update(type='MLP',input_dim=128,architecture='Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8)',hidden_dims=[64],output_dim=8,activation='ReLU',dropout=.2,trainable_in_this_run=True,probability_method='softmax')
    r['training'].update(regime='frozen_encoder_then_classifier',optimizer='AdamW',learning_rate=.001,batch_size=64,max_epochs=20,completed_epochs=20,early_stopping_patience=8,class_weighting='none',other_parameters={'weight_decay':1e-4,'gradient_clip':1.})
    r['selection'].update(best_epoch=14,selected_for_next_stage=False,selection_reason='Interface pilot, not part of the later 30-epoch fusion comparison')
    pilot_ck=torch.load(pilot_audio/'audio_pooled_mlp.pt',weights_only=True,map_location='cpu')
    pilot_norm=shared/'normalizers/audio_learned_pilot.json'
    write(pilot_norm,{'mean':pilot_ck['mean'].tolist(),'std':pilot_ck['std'].tolist()})
    r['preprocessing'].update(audio='Frozen learned-only M2 export',standardization_fit_split='train',augmentation_train_only=False,parameters={'normalizer_ref':ref(pilot_norm,dest),'normalizer_sha256':digest(pilot_norm)})
    r['pipeline']['classifier']['checkpoint_sha256']=digest(pilot_audio/'audio_pooled_mlp.pt')
    folder=records/rid;folder.mkdir(parents=True,exist_ok=True)
    with (pilot_audio/'validation_predictions.csv').open() as f:raw=list(csv.DictReader(f))
    keep=[x for x in raw if x['model']=='retrained_audio_pooled_mlp']
    temp=folder/'raw_predictions.csv'
    with temp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(keep[0]),lineterminator='\n');w.writeheader();w.writerows(keep)
    prediction_pair(r,folder,{'val':temp},{'val':old['retrained_audio_pooled_mlp']});temp.unlink()
    history=read(pilot_audio/'history.json')
    with (folder/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]),lineterminator='\n');w.writeheader();w.writerows(history)
    r['artifacts']['history_ref']=ref(folder/'history.csv',dest);r['artifacts']['checkpoint_sha256']=digest(pilot_audio/'audio_pooled_mlp.pt');asset('Audio pilot checkpoint',pilot_audio/'audio_pooled_mlp.pt')
    r['input_artifacts']=[{'role':'learned-only export','source_run_id':OLD_M2,'ref':None,'sha256':read(project/'data/m2_learned_only/import_audit.json')['archive_sha256']}]
    save(r,folder,'Validate learned-only audio interface before multimodal inputs were available.',
         '20-epoch single-seed pilot with no test evaluation. The original upstream M2 baseline is already recorded as '+OLD_M2+'; do not count it as another training run.')
    write(shared/'large_artifact_inventory.json',assets)
    write(shared/'export_inventory.json',{'published_on':'2026-10-20','new_records':len(all_records),
       'newly_trained_pipeline_records':26,'decision_configurations':2,'imported_upstream_reference_records':2,
       'reused_existing_M2_record':OLD_M2,'upstream_M2_checkpoint_original_sha256':'538cd18812f38cd4a28671a028dd376601363ff3799a01ad715f5fce209b6a6c',
       'notes':['Does not rerun experiments or change scores.','Synthetic checks are not counted as formal real-data records.',
                'Historical exports precede M2 round04 promotion; no current upstream default is silently substituted.',
                'Large inputs and M5 weights omitted; explicit inventory hashes and owner retrieval retained.']})
    make_summary(dest,upstream,all_records)
    make_report(shared,source,dest)
    # Copy exact current reproduction code/tests. Keep teammate-owned code/templates untouched.
    for name in ['models.py','feature_inputs.py','feature_pilot.py','study.py','study_diagnostics.py','study_report.py','final_evaluation.py']:
        dst=dest/'multimodal_fusion/fusion'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/'fusion'/name,dst)
    for name in ['test_feature_inputs.py','test_study_models.py','test_final_evaluation.py']:
        dst=dest/'multimodal_fusion/tests'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(project/'tests'/name,dst)
    print(f'Exported {len(all_records)} schema-v2 records to {dest}')


def digest_json(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def make_summary(dest,upstream,records):
    path=upstream/'experiment_records/summary.md';s=path.read_text()
    configs=[];results=[]
    for r in records:
        p=r['pipeline'];c=p['classifier'];f=p['fusion'];rid=r['run_id']
        configs.append('| ['+rid+']('+rid+'/run.json) | M5 / '+r['stage']+' | '+r['comparison_group']+' | '+('; '.join(x['method'] for x in p['feature_extractors']) or 'referenced upstream classifiers')+' | '+str(p['representation']['kind'])+' / '+str(p['representation']['output_shape'])+' | '+str(f['method'] or 'none')+' | '+str(c['architecture'] or c['type'])+' | '+r['training']['regime']+' | '+str(r['seed'] if r['seed'] is not None else 'N/A')+' |')
        for split in ('val','test'):
            m=r['metrics_by_split'][split]
            results.append('| '+rid+' | '+split+' | '+(str(m['n'])+' | '+' | '.join(f"{m[k]:.4f}" for k in ['macro_f1','accuracy','macro_precision','macro_recall'])+' | completed |' if m else '— | — | — | — | — | not_run |'))
    def append_table(s,heading,lines):
        start=s.index(heading);end=s.find('\n## ',start+len(heading));end=len(s) if end<0 else end
        chunk=s[start:end];matches=list(re.finditer(r'^\|.*\|$',chunk,re.M));assert matches
        pos=start+matches[-1].end()
        return s[:pos]+'\n'+'\n'.join(lines)+s[pos:]
    s=append_table(s,'## Real RAVDESS speech — pipeline configurations',configs)
    s=append_table(s,'## Real RAVDESS speech — full-pipeline results',results)
    selected=[r for r in records if r['pipeline']['fusion']['method']=='cross_attention' and r['comparison_group']==GROUP_FUSION]
    selection='; '.join('['+r['run_id']+']('+r['run_id']+'/run.json)' for r in selected)
    s=append_table(s,'## Validation-based selections',[
      f'| {GROUP_FUSION} | {selection} | primary fusion method, across all seeds | Selected cross_attention by validation mean0.6608; gated test0.6831 is descriptive and did not replace selection. |',
      '| M5_decision_fixed_exports_v1 | M5_fusion_decision_tuned_sNA_20261019_01 | audio probability weight | Validation-only alpha_audio=0.15; class order confirmed afterward without changing values. |'])
    s+='\n## M5 current results and limitations\n\n30 added M5 records =26 trained pipeline configurations/seeds +2 decision configurations +2 imported probability-head references. Existing M2 records are preserved. Means±sample SD are in the [M5 report](shared/M5/M5_RESULTS_20261020.md). Historical features correspond to the initial M2 checkpoint, not the newly promoted round04 default. Synthetic checks are excluded from these real-data counts. No test-score-based reselection was performed.\n'
    (dest/'experiment_records/summary.md').write_text(s)


def make_report(shared,source,dest):
    root=source/'multimodal_fusion/results';final=read(root/'m5_final_evaluation_20261019/results.json');summary=read(root/'m5_validation_study_20261019/default_summary.json')
    lines=['# M5 Fusion Results and Remaining Work','','Published: 2026-10-20. Historical frozen feature exports, fixed shared actor split. [All run records](../../summary.md).',
      '', '## Main results / 主要结果','','Mean across seeds42/43/44; sample SD uses ddof=1. Matched unimodal controls retrain an MLP on frozen features, and are distinct from the supplied original heads.','','| Pipeline | Validation macro-F1 | Test macro-F1 |','|---|---:|---:|']
    for method,s in final['summary'].items():lines.append(f"| {method} | {s['val']['macro_f1_mean']:.4f} ± {s['val']['macro_f1_std']:.4f} | {s['test']['macro_f1_mean']:.4f} ± {s['test']['macro_f1_std']:.4f} |")
    lines+=['','Cross-attention was selected by validation. Gated achieved the highest observed test mean, but test ranking was not used to change selection. Cross-attention did not preserve its validation advantage. / Cross-attention按验证集预选，门控测试较高是描述性结果，未据此改选模型。',
      '', '## Decision fusion / 决策融合','','| Pipeline | Audio weight | Validation macro-F1 | Test macro-F1 |','|---|---:|---:|---:|']
    for m,r in final['decision'].items():lines.append(f"| {m} | {r['alpha_audio']:.2f} | {r['val']['macro_f1']:.4f} | {r['test']['macro_f1']:.4f} |")
    lines+=['','Visual probability class order was explicitly confirmed by the teammate via the user after initial evaluation; input checksum and saved probability values were verified unchanged. No permutation search or metric rerun was used. The original visual head test F1=0.2150 is distinct from the retrained visual MLP test mean0.3918. / 类别顺序已确认，当前重点是原视觉分类头、导出来源与泛化表现，而非猜列排列。',
      '', '## Interpretation / 解释分析','','- Fusion uses complementary information: zeroing either modality reduces validation performance; wrong visual pairing especially harms attention. This is feature-input sensitivity, not causal attribution.',
      '- Average repair/harm counts relative to the validation-preselected visual control: concat52.33/26.00, gated50.33/30.67, cross_attention48.00/18.33 per seed.',
      '- Learned-attention validation mean0.6608 versus uniform-attention0.6627: do not claim that learned temporal alignment explains the gain.',
      '- Gate noise responses do not demonstrate reliable-modality switching. Gate means are not contribution percentages.',
      '- Test actor17 visual control F1=0.0623, audio control0.5602; actor effects warrant preprocessing and provenance checks, but do not prove the cause.',
      '- Targeted hidden32/dropout0.4 trials did not meet the prespecified0.02 validation improvement gate for more seeds. The isolated best tuning score0.6828 is not promoted as the main result.',
      '', '![Validation versus test](figures/final/validation_test_comparison.png)','','![Validation defaults](figures/validation/validation_summary.png)',
      '', '## Input and reproduction / 输入与复现','','- Audio pooled176=48 handcrafted+128 learned; attention audio sequence128D. Visual pooled576 and16×576 sequence. Cross-attention and pooled baselines differ in representation and parameter count.',
      '- Historical audio source matches the original checkpoint associated with M2_encoder_transformer_s42_20261015_00. M2 has since promoted another encoder; these results do not use that new export.',
      '- Class order: neutral, calm, happy, sad, angry, fearful, disgust, surprise. Actor split is shared16/4/4. Visual16 tokens are assumed valid pending upstream provenance.',
      '- Unknown M1/M4 run IDs, upstream classifier architecture and calibration metadata remain null. M5 checkpoint weights and raw features are not uploaded; see [artifact inventory](large_artifact_inventory.json) for hashes and retrieval status.',
      '- Source snapshots, patch, input manifest, per-run predictions and histories permit review. Run the reproduction code only after obtaining the exact feature version; do not silently use the newest M2 default.',
      '', '```bash','cd multimodal_fusion','python -m unittest discover -s tests -v','python -m fusion.feature_pilot --output results/new_pilot --seed 42',
      'python -m fusion.study --pilot results/new_pilot --output results/new_study',
      'python -m fusion.study_diagnostics --study results/new_study --pilot results/new_pilot','```',
      '', '## Remaining work / 尚未完成','','1. Confirm original visual head/checkpoint/export identity and preprocessing; the class order itself is confirmed.',
      '2. Obtain final M3 A/B/A+B visual selection; current results use a learned-only MobileNet candidate.',
      '3. Complete fixed-head audio48/128/176 and handcrafted/learned/combined visual representation ablations.',
      '4. Parameter-matched projection/residual controls and uniform-attention retraining to isolate attention selection effects.',
      '5. Raw audio/video degradation and external manually labelled samples/demo. Current noise experiments are feature-level only.',
      '', 'The test set is already exposed. Any subsequent test-informed model change must be recorded as exploration, not development on an unseen test set. / 测试集已查看，后续改方案须披露测试暴露。',
      '', '## Validation / 校验','','Publication validator recomputes fixed-eight-class metrics from every prediction CSV, checks summary rows against run.json, verifies shared split/classes/IDs and preserves existing M2 records. Unknown values are not guessed. Synthetic engineering tests are not counted as formal RAVDESS training runs.']
    (shared/'M5_RESULTS_20261020.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--dest',required=True);p.add_argument('--upstream',required=True)
    a=p.parse_args();export(a.source,a.dest,a.upstream)
