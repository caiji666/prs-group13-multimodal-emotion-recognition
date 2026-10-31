"""Export the external pilot using the team's v2 experiment record format."""
import copy
import csv
import json
from pathlib import Path
import platform
import subprocess
import zipfile

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from .engine import Engine, CLASSES, NAMES
from .external_eval import OUT, ROOT, sha, save
from .extraction import ADAPTER, ADAPTER_SHA
from .setup_assets import ASSETS, WEIGHT_SHA


def main():
    plan=json.loads((OUT/'plan.json').read_text());summary=json.loads((OUT/'summary.json').read_text())
    lock=json.loads((OUT/'model_lock.json').read_text());engine=Engine()
    predictions=list(csv.DictReader((OUT/'predictions.csv').open()))
    assert len(predictions)==264 and len({(r['sample_id'],r['method']) for r in predictions})==264
    assert summary['successful']==24 and not summary['failed']
    project=ROOT.parent;template=json.loads((project/'docs/templates/run.template.json').read_text())
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip()
    snapshot=OUT/'code_snapshot.zip'
    files=list(ROOT.rglob('*.py'))+[ROOT/'web/index.html',ROOT/'web/app.js',ROOT/'web/style.css']+list((project/'multimodal_fusion/fusion').glob('*.py'))
    files=[f for f in files if '.venv' not in f.parts and 'test-artifacts' not in f.parts and 'sessions' not in f.parts]
    with zipfile.ZipFile(snapshot,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(files):archive.write(f,f.relative_to(project))
    def ref(path):return str(Path(path).relative_to(project))
    six=sorted(CLASSES.index(c) for c in plan['label_mapping'].values());records=[];lines=[]
    for method,m in summary['metrics'].items():
        rid=f'M5_external_cremad_{method}_s42_20261027_01';folder=OUT/'records'/rid;folder.mkdir(parents=True,exist_ok=True)
        row=engine.row(method) if method not in ('upstream_audio','decision_equal','decision_tuned') else None
        ck=torch.load(Path(row['folder'])/'checkpoint.pt',map_location='cpu',weights_only=True) if row else None
        r=copy.deepcopy(template)
        r.update(run_id=rid,member='M5',owner='Jiang Yushan',stage='fusion' if method not in ('upstream_audio','audio_only','visual_only') else 'classifier',
                 comparison_group='M5_cremad_frozen_raw_video_pilot_v1',pipeline_name=NAMES.get(method,'Original M2 WavLM audio head'),
                 run_status='completed',test_status='not_run',external_status='completed',started_at=plan['created_utc'],finished_at=summary['completed_utc'],seed=42,
                 dataset_id='crema_d_DFA_1001_1004_pilot_v1',data_kind='external',split_ref=ref(OUT/'plan.json'),split_sha256=sha(OUT/'plan.json'),
                 input_manifest_ref=ref(OUT/'media_manifest.json'),input_manifest_sha256=sha(OUT/'media_manifest.json'),
                 counts={'expected':{'external':24},'actual':{'external':24}},excluded_samples={'count':0,'id_reason_manifest_ref':None,'handling_policy':plan['failure_policy']},
                 git_commit=git_commit,working_tree_dirty=True,code_patch_ref=ref(snapshot),command='demo/.venv/bin/python -m demo.external_eval run',
                 environment={'python':platform.python_version(),'framework_and_version':'torch '+torch.__version__,'device':'CPU; MediaPipe local GL context','dependencies_ref':ref(ROOT/'runtime_versions.json')})
        r['source_runs']['audio_learned']='M2_encoder_wavlm_refinement_s42_20261021_05'
        r['source_runs']['visual_final']=engine.row('visual_only')['run_id'] if method!='upstream_audio' else None
        r['input_artifacts']=[{'ref':ref(OUT/'model_lock.json'),'sha256':sha(OUT/'model_lock.json')},
                              {'ref':ref(ADAPTER),'sha256':ADAPTER_SHA},{'ref':ref(ASSETS/'pytorch_model.bin'),'sha256':WEIGHT_SHA},
                              {'ref':ref(snapshot),'sha256':sha(snapshot)}]
        r['preprocessing'].update(audio='Mono 16kHz; librosa trim top_db35; WavLM layer6 fp16 round-trip; frozen temporal adapter and attentive pooling.',
                                  visual='16 uniform frames; MediaPipe detect/eye align; Haar DWT HC320; same extractor as UI.',
                                  standardization_fit_split='original RAVDESS train only; no external fitting',augmentation_train_only=None)
        r['pipeline']['feature_extractors']=[
            {'component_id':'audio','modality':'audio','kind':'learned','method':'WavLM base plus layer6 + M2 adapter','checkpoint_ref':ref(ADAPTER),'checkpoint_sha256':ADAPTER_SHA,'trainable_in_this_run':False,'output_sequence_shape':'[1,T,128]','output_pooled_shape':'[1,128]'},
            {'component_id':'visual','modality':'visual','kind':'handcrafted','method':'Haar DWT','trainable_in_this_run':False,'output_sequence_shape':'[1,16,320]','output_pooled_shape':'[1,320]','parameters':{'provenance_ref':ref(ROOT/'vendor/manifest.json')}}]
        if method=='upstream_audio':r['pipeline']['feature_extractors']=r['pipeline']['feature_extractors'][:1]
        elif method=='visual_only':r['pipeline']['feature_extractors']=r['pipeline']['feature_extractors'][1:]
        r['pipeline']['representation'].update(kind='frozen extracted features / probabilities',components=[x['component_id'] for x in r['pipeline']['feature_extractors']],normalization='saved checkpoint normalizers; no external fitting',normalization_fit_split='original RAVDESS train')
        r['pipeline']['fusion'].update(applied=method not in ('upstream_audio','audio_only','visual_only'),method=method,trainable_in_this_run=False,
                    input_source_runs=r['source_runs'],parameters={'checkpoint_config':ck['config'] if ck else None,'alpha_audio':.7 if method=='decision_tuned' else .5 if method=='decision_equal' else None})
        head=r['pipeline']['classifier'];head.update(type='none' if method.startswith('decision') else 'frozen neural head',source_run_id=row['run_id'] if row else 'M2_encoder_wavlm_refinement_s42_20261021_05' if method=='upstream_audio' else None,
            architecture='Defined by the referenced frozen checkpoint config/state_dict and code snapshot.',trainable_in_this_run=False,probability_method='weighted sum of frozen softmax heads' if method.startswith('decision') else 'softmax',
            checkpoint_ref=ref(Path(row['folder'])/'checkpoint.pt') if row else ref(ADAPTER) if method=='upstream_audio' else None,
            checkpoint_sha256=lock['checkpoints'][method] if row else ADAPTER_SHA if method=='upstream_audio' else None)
        r['training']['regime']='inference only; no parameter update or normalizer fitting'
        r['selection'].update(metric=None,split=None,selected_for_next_stage=False,selection_reason='All methods fixed before external inference. No external-score model selection.',config_locked_at=plan['created_utc'])
        rr=[v for v in predictions if v['method']==method];y=np.array([int(v['y_true']) for v in rr]);p=np.array([int(v['y_pred']) for v in rr])
        prec,rec,f1,support=precision_recall_fscore_support(y,p,labels=list(range(8)),zero_division=0)
        metrics={'n':24,'accuracy':m['accuracy_all_planned'],'macro_precision':float(prec[six].mean()),'macro_recall':float(rec[six].mean()),'macro_f1':m['macro_f1_six_reference_classes'],
                 'macro_labels':six,'prediction_labels':list(range(8)),'macro_f1_eight_classes_absent_as_zero':m['macro_f1_eight_classes_absent_as_zero'],
                 'per_class':[{'label':i,'class_name':c,'precision':float(prec[i]),'recall':float(rec[i]),'f1':float(f1[i]),'support':int(support[i])} for i,c in enumerate(CLASSES)],'confusion_matrix_counts':m['confusion_matrix_eight_classes']}
        r['metrics_by_split']={'val':None,'test':None,'external':metrics}
        r['artifacts'].update(original_metrics_ref=ref(OUT/'summary.json'),predictions_external_ref=ref(folder/'predictions_external.csv'),checkpoint_ref=head['checkpoint_ref'],checkpoint_sha256=head['checkpoint_sha256'],code_snapshot_ref=ref(snapshot))
        r['notes_ref']=ref(OUT/'REPORT.md');save(folder/'run.json',r)
        fields=['run_id','dataset_id','split','sample_id','actor_id','y_true','y_pred']+['p_'+c for c in CLASSES]
        with (folder/'predictions_external.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for v in rr:writer.writerow({'run_id':rid,'dataset_id':r['dataset_id'],'split':'external',**{k:v[k] for k in fields if k in v}})
        records.append({'run_id':rid,'method':method,'ref':ref(folder/'run.json')})
        lines.append(f"| {NAMES.get(method,'Original M2 WavLM audio head')} | {m['correct']}/24 | {m['accuracy_all_planned']:.2%} | {m['macro_f1_six_reference_classes']:.4f} | {m['macro_f1_eight_classes_absent_as_zero']:.4f} |")
    save(OUT/'record_index.json',records)
    table='\n'.join(lines)
    report=f'''# External-video generalization pilot — 2026-10-27

## Conclusion / 结论

The raw-video system processes independent CREMA-D recordings successfully, but this pilot does **not** establish reliable external generalization. All **24/24** clips completed extraction and inference; faces were detected in **384/384** sampled frames. The predeclared primary fusion model, concatenation, achieved **9/24 = 37.5% accuracy**, with six-reference-class Macro-F1 **0.3346**. The original M2 audio head achieved **10/24 = 41.7%**, Macro-F1 **0.4076**. Fusion did not consistently outperform this audio baseline.

系统已经能够处理 RAVDESS 之外的真人视频；但“可以输出预测”与“跨数据集识别可靠”不同。这批结果提示当前模型存在明显的迁移局限，不能据此宣称适用于任意真实场景。当前尚无法把领域差异与原始视频预处理复现误差完全区分。

## Frozen protocol

- Source: [official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D), revision `{plan['revision']}`. Source README and license are retained alongside this report. The public repository requests that users register usage via its linked form; no personal information was submitted on your behalf.
- Selection was written to [plan.json](plan.json) **before inference**: actors 1001–1004, sentence DFA, intensity XX, all six emotion codes. This is a deterministic convenience sample, not a random representative benchmark. Each emotion has four recordings.
- Reference labels are the official filenames’ **intended/performed emotions**, not crowd-majority ratings or verified internal emotions. Mapping: ANG→angry, DIS→disgust, FEA→fearful, HAP→happy, NEU→neutral, SAD→sad.
- All models, normalizers, checkpoints and seed 42 were fixed. No training, external-label fitting, probability recalibration, threshold search, or RAVDESS test re-evaluation was performed. Model hashes and the validation-only decision-weight grid are in [model_lock.json](model_lock.json).
- The official FLV bytes are checked against the pinned Git LFS SHA256 and decoded directly for scoring. MP4 copies are only for browser upload checks; that separate lossy-transcoded UI example is not mixed into these metrics.
- Audio: original WavLM backbone + M2 adapter, 128D pooled and variable-length 128D sequence. Visual: 16 MediaPipe-aligned faces → Haar DWT 320D per frame and temporal mean. All models use the same extracted inputs.
- **Raw visual probabilities use the M5 visual-only MLP**, because the upstream visual SVM checkpoint has not been delivered. Validation-weighted decision fusion has fixed audio weight **0.70**, learned previously from RAVDESS validation predictions with this MLP. It is a separate UI protocol from the paper’s audio+SVM fusion (0.60).
- Predictions remain **eight-class argmax**. Calm and surprise are not masked and probabilities are not renormalized. They count as errors when predicted for these six-label samples.
- Primary Macro-F1 averages the six reference classes. The secondary eight-class figure assigns zero F1 to the two absent reference classes. These metrics must not be compared directly with the RAVDESS eight-class benchmark as if they were the same evaluation.
- No failures or exclusions. All planned clips contribute to accuracy. Eleven fixed methods yield **264 sample-method predictions**.

## Results

| Frozen pipeline (seed 42) | Correct | Accuracy | Macro-F1, six references | Macro-F1, eight with absent=0 |
|---|---:|---:|---:|---:|
{table}

LMF has the highest descriptive score on this small sample (11 correct); this is **not** a new validation-based selection or evidence of a statistically reliable advantage. One clip changes accuracy by 4.17 percentage points. The 24 clips are clustered within just four actors; no independent-clip confidence interval is claimed.

## What the errors tell us

1. **Detection and transport are working.** Every sample completed real decoding, audio encoding, face detection, handcrafted extraction and inference. This excludes gross pipeline failure for this batch; it does not prove precise parity with teammates’ exported RAVDESS features.
2. **Errors affect both modalities.** Original audio obtains 10 correct and the visual MLP 6. Concatenation and gating both obtain 9, with the same predicted class on every pilot clip. Happy and neutral repeatedly become fearful/sad. Cross-attention predicts sad for 16 clips and calm for five; a normalized attention heatmap cannot establish recognition quality.
3. **High probability can be wrong.** Five incorrect original-audio predictions exceed 0.90 maximum probability; concatenation has two. Probabilities shown in the UI are uncalibrated model outputs, not accuracy estimates.
4. **Domain shift is plausible, not isolated as the sole cause.** Actors, utterances, recording conditions and emotion expression differ from RAVDESS. Label interpretation also differs, and the UI substitutes a visual MLP for the missing original SVM. A raw-RAVDESS parity check is still needed to separate preprocessing mismatch from domain transfer.

## Next experiments, in order

1. Obtain a small set of original RAVDESS videos matching delivered sample IDs. Re-extract with this pipeline; compare pooled/sequence features, original audio probabilities and frozen fusion predictions against the saved exports. Resolve any mismatch before attributing all external errors to generalization.
2. Obtain the final visual SVM artifact and exact preprocessing from M3, then evaluate that **new, explicitly versioned** UI configuration. Keep these MLP-based results unchanged.
3. Freeze a larger cross-dataset protocol covering more actors and all 12 CREMA-D sentences, with documented intended-vs-crowd-label policy and actor-level uncertainty. Do not choose samples based on these predictions. Keep the new evaluation separate from RAVDESS test.
4. If training adaptations, split external actors into adaptation/development/held-out sets first. Compare unchanged checkpoints, calibration, and domain adaptation on the untouched external holdout. Never tune and report on the same 24 clips.
5. Separately assess realistic conditions: spontaneous speech, languages, noise/music, multiple faces and multiple speakers. Add input-quality/unsupported-input handling; validate any rejection rule on development data first. This pilot covers acted English single-speaker clips only.

## Reproduction and artifacts

```bash
demo/.venv/bin/python -m demo.external_eval prepare
demo/.venv/bin/python -m demo.external_eval run
demo/.venv/bin/python -m demo.external_report
```

- [Raw prediction table](predictions.csv): all 264 predictions with reference labels and eight probability columns.
- [Structured summary](summary.json): per-class F1, full 8×8 confusion matrices, failure counts, face coverage and incorrect-prediction probability summaries.
- [Media provenance](media_manifest.json): official URLs, fixed revision, byte counts and SHA256.
- [Team v2 record index](record_index.json): one frozen pipeline × seed per run; `data_kind=external`, separate `metrics_by_split.external` and `predictions_external.csv`. RAVDESS val/test fields remain null because this run evaluates neither.
- [Code snapshot](code_snapshot.zip): the dirty working-tree source used by this pilot, including the frozen fusion implementation; checkpoint hashes are recorded separately. Large weights and raw videos remain local and gitignored.
- Local media: `demo/test-artifacts/crema_d/`; each sample folder contains extracted NPZ, metadata, face crops and per-method explanation JSON. Source videos are not published by this task. CREMA-D ODbL/Database Contents License notices are retained here; follow them if redistributing data or derived databases.

The English UI was verified through a real external MP4 upload, and its gate explanation, modality comparison and uncalibrated-probability labels were inspected. Ten model regression tests and eleven HTTP checks passed after translation.
'''
    (OUT/'REPORT.md').write_text(report)
    print('Exported 11 v2 records and REPORT.md; 264 unique predictions verified.')

if __name__=='__main__':main()
