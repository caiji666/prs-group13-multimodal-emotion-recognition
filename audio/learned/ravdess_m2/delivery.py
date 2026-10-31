"""Build portable model and feature deliveries for audio and fusion teammates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import torch

from .data import LABELS
from .feature_export import verify_source
from .feature_io import validate_bundle
from .inference import AudioInference
from .fusion_export import export_learned_audio


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build_delivery(features, output, make_zip=False):
    features=Path(features).resolve(); output=Path(output).resolve()
    archive=Path(str(output)+'.zip')
    if output.exists() or (make_zip and archive.exists()):
        raise FileExistsError('Use a new delivery directory and ZIP filename')
    if output.is_relative_to(features) or features.is_relative_to(output):
        raise ValueError('Delivery and source features must not contain each other')
    validation=validate_bundle(features)
    if json.loads((features/'status.json').read_text()).get('status')!='completed':
        raise ValueError('Stage 4 export is incomplete')
    metadata=json.loads((features/'metadata.json').read_text())
    selection=json.loads((features/'provenance/stage3_comparison.json').read_text())
    protocol=json.loads((features/'provenance/stage3_protocol.json').read_text())
    if selection.get('diagnostic') or selection.get('selection_metric')!='validation_macro_f1' or selection.get('test_evaluated'):
        raise ValueError('Require formal validation-only selection')
    chosen=[r for r in selection['comparison'] if r['encoder']==selection['selected_encoder']]
    if len(chosen)!=1 or chosen[0]['macro_f1']!=max(r['macro_f1'] for r in selection['comparison']):
        raise ValueError('Selected encoder is not the validation winner')
    if metadata['encoder']!=chosen[0]['encoder'] or metadata['best_epoch']!=chosen[0]['best_epoch']:
        raise ValueError('Selected encoder/epoch differs from feature bundle')
    saved=torch.load(features/'checkpoint/best.pt',map_location='cpu',weights_only=True)
    if saved['provenance'].get('diagnostic') or saved['class_names']!=LABELS or saved['encoder']!=metadata['encoder'] or saved['epoch']!=metadata['best_epoch']:
        raise ValueError('Checkpoint metadata differs from selection')
    if saved['preprocessing']!=json.loads((features/'preprocessing.json').read_text()) or saved['provenance']['split_sha256']!=metadata['split_sha256']:
        raise ValueError('Checkpoint preprocessing/splits differ from export')
    for name in ('models.py','data.py'):
        verify_source(protocol,name)
    # Strictly load the checkpoint before creating any deliverable.
    inference=AudioInference(features/'checkpoint/best.pt')
    output.mkdir(parents=True,exist_ok=False)
    try:
        shutil.copytree(features,output/'features',ignore=shutil.ignore_patterns('__pycache__'))
        m5_metadata=export_learned_audio(features,inference.model,output/'m5_learned_feature_only')
        shutil.copy2(features/'checkpoint/best.pt',output/'best_audio_encoder.pt')
        shutil.copy2(features/'actor_splits.csv',output/'actor_splits.csv')
        shutil.copy2(features/'manifest.csv',output/'audio_features_index.csv')
        shutil.copy2(features/'provenance/stage3_comparison.csv',output/'audio_model_comparison.csv')
        shutil.copy2(features/'provenance/stage3_comparison.json',output/'audio_model_comparison.json')
        package=output/'ravdess_m2'; package.mkdir()
        for name in ('__init__.py','models.py','data.py','feature_io.py','inference.py'):
            shutil.copy2(Path(__file__).with_name(name),package/name)
        (output/'audio_inference.py').write_text('from ravdess_m2.inference import main\n\nif __name__ == "__main__":\n    main()\n')
        project=Path(__file__).resolve().parents[1]
        shutil.copy2(project/'requirements.txt',output/'requirements.txt')
        shutil.copy2(project/'requirements-tested.txt',output/'requirements-tested.txt')
        config=dict(encoder=metadata['encoder'],best_epoch=metadata['best_epoch'],architecture=dict(input_dim=saved['preprocessing']['n_mfcc'],hidden_dim=metadata['feature_dim'],num_classes=8),
                    preprocessing=saved['preprocessing'],inference_augmentation=False,class_names=LABELS,
                    label_mapping={str(i):name for i,name in enumerate(LABELS)},normalization=dict(source='checkpoint buffers; fitted on clean train frames only',
                    mean=saved['model_state_dict']['feature_mean'].tolist(),std=saved['model_state_dict']['feature_std'].tolist()),
                    checkpoint_sha256=metadata['checkpoint_sha256'],split_sha256=metadata['split_sha256'],
                    sequence_shape='[T,128]',pooled_shape='[128]',padding_mask='True means padding',
                    audio_features_format='features/per_sample/{split}/{sample_id}.npz plus pooled NPZ/CSV; no redundant audio_features.pt')
        write_json(output/'audio_config.json',config)
        contract=dict(m2_status='complete',audio_module_status='awaiting_M1_final_classifier_and_representation',
                      join_keys=['sample_id'],required_metadata=['split','label'],class_names=LABELS,
                      outputs=dict(F_seq=dict(owner='M2',ready=True,consumer='M5',shape='[T,128]',location='features/per_sample/'),
                                   F_sequence=dict(owner='M2',ready=True,consumer='M5',shape='[T,128]',location='m5_learned_feature_only/per_sample/'),
                                   F_audio_learned_feature_only=dict(owner='M2',ready=True,consumer='M5',shape='[128]',location='m5_learned_feature_only/',purpose='functional testing; no handcrafted fusion'),
                                   P_audio_learned_feature_only=dict(owner='M2',ready=True,consumer='M5',shape='[8]',location='m5_learned_feature_only/',purpose='functional testing; selected M2 classifier probabilities'),
                                   F_learned=dict(owner='M2',ready=True,consumer='M1',shape='[128]',location='features/pooled/'),
                                   F_audio=dict(owner='M1',ready=False,consumer='M5',shape='[H+128] for handcrafted+learned configuration; H determined by M1'),
                                   P_audio=dict(owner='M1',ready=False,consumer='M5',shape='[8]',selection='M1 Level-2 validation comparison determines final audio classifier')),
                      m5_testing_interface='ready: F_sequence, F_audio_learned_feature_only, P_audio_learned_feature_only',
                      inference_probabilities='M2 learned-only classifier; not automatically final P_audio',
                      test_usage='Features available; reserve test task metrics for final evaluation')
        write_json(output/'stage5_contract.json',contract)
        readme=f'''# M2 Stage 5/6 delivery\n\nSelected encoder: {metadata['encoder']}, epoch {metadata['best_epoch']}. Validation-selected; no test model selection.\n\nM1: read `features/pooled/{{train,val,test}}.npz` or CSV for F_learned. M5: read `m5_learned_feature_only/per_sample/` for F_sequence, F_audio_learned_feature_only and P_audio_learned_feature_only, plus length/mask. Split-level pooled/probability NPZ and CSV are in `m5_learned_feature_only/pooled/`. This learned-only testing interface has no handcrafted fusion. Match sample_id and verify label/split; never pair by row position. `features/README.md` (when present) and `features/handoff_reader.py` describe/read the Stage 4 interface.\n\n`stage5_contract.json` distinguishes completed M2 outputs from M1's pending final F_audio/P_audio. `best_audio_encoder.pt`, `audio_config.json`, `audio_model_comparison.csv` and `audio_features_index.csv` provide checkpoint/config/results/index. All sample features remain in portable NPZ files rather than a duplicate giant audio_features.pt.\n\n## Inference on a new recording\n\nFrom this extracted delivery folder (Python 3.9–3.12):\n\n```bash\npython3 -m venv .venv\n.venv/bin/python -m pip install -r requirements.txt\n.venv/bin/python audio_inference.py --audio /absolute/path/new_audio.wav --external --output inference/new_audio.npz\n```\n\nThe checkpoint defaults to `best_audio_encoder.pt`. A valid RAVDESS filename automatically uses the enclosed actor_splits.csv when --external is absent. External samples use a content-derived ID, split=external, label=-1, actor_id=-1; no ground-truth emotion is invented. Outputs include sequence, pooled, lengths/mask, logits/probabilities and class order. These probabilities are from M2's learned-only head, **not M1's final P_audio**. No augmentation, fitting or training is performed. The checkpoint's preprocessing and training-fitted normalization are always restored. Existing output files are rejected.\n\nDo not infer video-frame alignment from MFCC indices. Only fit downstream scalers/models on train; select with val and reserve test for final evaluation. M1 handcrafted extraction/concatenation, M1 final classifier and M5 fusion are not included.\n'''
        (output/'README.md').write_text(readme)
        files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
               for p in output.rglob('*') if p.is_file()}
        write_json(output/'delivery_manifest.json',dict(files_sha256=files,stage4_validation=validation,
                                                      m2_complete=True,m1_final_audio_outputs_included=False,m5_learned_only_interface=m5_metadata))
        if make_zip:
            temporary=archive.with_suffix('.zip.partial')
            try:
                with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                    for path in sorted(output.rglob('*')):
                        if path.is_file(): z.write(path,Path(output.name)/path.relative_to(output))
                with zipfile.ZipFile(temporary) as z:
                    if z.testzip() is not None: raise ValueError('ZIP integrity failed')
                temporary.rename(archive)
            finally:
                if temporary.exists(): temporary.unlink()
        print(f'Delivery ready: {output}'+(f'\nZIP ready: {archive}' if make_zip else ''))
        return output
    except BaseException:
        # Preserve partial files for diagnosis, never present them as a complete delivery.
        (output/'DELIVERY_FAILED.txt').write_text('Packaging failed; do not hand off this incomplete directory. Use a new output directory when retrying.\n')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',required=True,help='Completed Stage 4 bundle')
    parser.add_argument('--output',required=True,help='New delivery directory')
    parser.add_argument('--zip',action='store_true',help='Also create a ZIP beside the directory')
    args=parser.parse_args()
    torch.set_num_threads(2)
    build_delivery(args.features,args.output,args.zip)

if __name__=='__main__': main()
