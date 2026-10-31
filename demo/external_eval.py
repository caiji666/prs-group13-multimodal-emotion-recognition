"""Frozen, small CREMA-D cross-dataset pilot. Never tunes on external labels.

Run prepare first (network), then run (local inference). Videos stay gitignored.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'external_evaluation_20261027'
MEDIA = ROOT/'test-artifacts/crema_d'
REVISION = '1658cd342dff90010aa843eaeebd53610a08b1dc'
REPO = 'CheyneyComputerScience/CREMA-D'
EMOTIONS = {'ANG':'angry','DIS':'disgust','FEA':'fearful','HAP':'happy','NEU':'neutral','SAD':'sad'}
METHODS = ['audio_only','visual_only','concat','gated','cross_attention','cross_bidirectional',
           'audio_residual_aux','lmf_rank4','decision_equal','decision_tuned']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'PRS-M5-research-pilot'}),timeout=90) as response:
        return response.read()

def prepare():
    OUT.mkdir(exist_ok=True); MEDIA.mkdir(parents=True,exist_ok=True)
    plan_path=OUT/'plan.json'
    plan={'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':'CREMA-D',
          'repository':'https://github.com/'+REPO,'revision':REVISION,
          'selection':'First four actor IDs 1001–1004; sentence DFA; all six emotions; intensity XX. Fixed before inference; no selection based on predictions.',
          'sample_ids':[f'{actor}_DFA_{emotion}_XX' for actor in range(1001,1005) for emotion in EMOTIONS],
          'label_mapping':EMOTIONS,'reference':'Intended/performed emotion in the official filename; not crowd majority or a claim about true internal emotion.',
          'methods':['upstream_audio']+METHODS,'primary_method':'concat','seed':42,
          'preprocessing':'Same local raw-video extractor as UI: MediaPipe aligned HC320 and WavLM128; no external fitting.',
          'decision_audio_weight':0.70,'decision_weight_source':'Existing RAVDESS validation-only M5 visual MLP protocol; not the report SVM protocol.',
          'prediction_space':'All eight RAVDESS classes; never mask calm/surprise or renormalize to six classes.',
          'metric':'Accuracy and Macro-F1 over the six reference classes; predictions outside these classes count as errors. Also report eight-class Macro-F1 with absent-class F1=0.',
          'failure_policy':'Keep every planned sample. Report failures; no silent replacements. Accuracy over all planned samples counts extraction failures as wrong.',
          'limitations':['24 clips, four actors and one sentence: feasibility pilot only, not a representative benchmark.',
                         'Both datasets contain acted English speech; this does not validate spontaneous speech, other languages, background music or multiple speakers.',
                         'Raw visual probabilities use the M5 MLP because the delivered SVM checkpoint is unavailable.',
                         'No raw-RAVDESS re-extraction parity check yet: preprocessing mismatch and domain shift are not disentangled.',
                         'No claim that foundation-model pretraining excluded all CREMA-D material.']}
    if not plan_path.exists():save(plan_path,plan)
    else:
        old=json.loads(plan_path.read_text())
        assert old['sample_ids']==plan['sample_ids'] and old['methods']==plan['methods']
        plan=old
    for name in ['README.md','LICENSE.txt']:
        p=OUT/('CREMA_D_'+name)
        if not p.exists():p.write_bytes(fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/{name}'))
    def download(sid):
        target=MEDIA/(sid+'.flv')
        url=f'https://media.githubusercontent.com/media/{REPO}/{REVISION}/VideoFlash/{sid}.flv'
        pointer=fetch(f'https://raw.githubusercontent.com/{REPO}/{REVISION}/VideoFlash/{sid}.flv').decode()
        oid=next(line.split(':',1)[1] for line in pointer.splitlines() if line.startswith('oid sha256:'))
        size=int(next(line.split()[1] for line in pointer.splitlines() if line.startswith('size ')))
        if not target.exists():target.write_bytes(fetch(url))
        assert target.stat().st_size==size and sha(target)==oid,sid
        print('Verified download '+sid,flush=True)
        return {'sample_id':sid,'url':url,'sha256':oid,'bytes':size,'file':str(target.relative_to(ROOT.parent))}
    with ThreadPoolExecutor(max_workers=4) as pool: manifest=list(pool.map(download,plan['sample_ids']))
    save(OUT/'media_manifest.json',manifest)
    print('Prepared 24 hash-verified clips; no predictions inspected.',flush=True)

def run():
    import imageio_ffmpeg
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from .engine import Engine,CLASSES,clean
    from .extraction import extract_video
    plan=json.loads((OUT/'plan.json').read_text())
    manifest=json.loads((OUT/'media_manifest.json').read_text());assert len(manifest)==24
    engine=Engine(); alpha=engine.raw_visual_alpha()
    assert alpha['alpha']==plan['decision_audio_weight'] and not alpha['test_used']
    # Resolve and hash every frozen model before the first external prediction.
    locks={method:engine.load(method)[3] for method in METHODS if not method.startswith('decision')}
    save(OUT/'model_lock.json',{'checkpoints':locks,'decision_protocol':alpha,'plan_sha256':sha(OUT/'plan.json')})
    rows=[];failed=[];quality=[]
    for item in manifest:
        sid=item['sample_id'];truth=EMOTIONS[sid.split('_')[2]];label=CLASSES.index(truth)
        original=ROOT.parent/item['file'];assert sha(original)==item['sha256']
        folder=MEDIA/sid;folder.mkdir(exist_ok=True)
        video=folder/'input.mp4';start=time.monotonic()
        try:
            # Only the browser-compatible copy is transcoded; the frozen inference
            # path consumes the original FLV to avoid changing audio before evaluation.
            if not video.exists():
                subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-nostdin','-i',str(original),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart','-y',str(video)],check=True,timeout=60)
            arrays,meta=extract_video(original,folder,face_mode='mediapipe',audio_representation='learned')
            save(folder/'input_metadata.json',meta)
            quality.append({'sample_id':sid,'frames_found':meta['frames_found'],'duration_sec':meta['duration_sec'],'warnings':meta['warnings']})
            predictions={'upstream_audio':arrays['audio_probs'][0].tolist()}
            for method in METHODS:
                result=engine.predict(arrays,meta,method)
                result['scope']='frozen CREMA-D pilot; labels used only after inference for scoring'
                save(folder/(method+'.json'),result);predictions[method]=result['probabilities']
            for method,prob in predictions.items():
                assert np.isfinite(prob).all() and abs(sum(prob)-1)<1e-5
                pred=int(np.argmax(prob))
                rows.append({'sample_id':sid,'actor_id':sid.split('_')[0],'method':method,'reference':truth,'y_true':label,
                             'prediction':CLASSES[pred],'y_pred':pred,'correct':pred==label,'max_probability':max(prob),
                             **{'p_'+c:prob[i] for i,c in enumerate(CLASSES)}})
            print(f'Completed {sid}: {time.monotonic()-start:.1f}s, faces {meta["frames_found"]}/16',flush=True)
        except Exception as exc:
            failed.append({'sample_id':sid,'reference':truth,'error':str(exc)})
            print(f'FAILED {sid}: {exc}',flush=True)
        save(OUT/'progress.json',{'completed_clips':len(quality),'failed':failed,'planned':24})
    with (OUT/'predictions.csv').open('w',newline='') as f:
        if rows:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    six=sorted(CLASSES.index(c) for c in EMOTIONS.values()); metrics={}
    for method in plan['methods']:
        rr=[r for r in rows if r['method']==method]; y=[r['y_true'] for r in rr];p=[r['y_pred'] for r in rr]
        if not rr:continue
        metrics[method]={'n_successful':len(rr),'correct':sum(r['correct'] for r in rr),
                         'accuracy_successful':accuracy_score(y,p),'accuracy_all_planned':sum(r['correct'] for r in rr)/24,
                         'macro_f1_six_reference_classes':f1_score(y,p,labels=six,average='macro',zero_division=0),
                         'macro_f1_eight_classes_absent_as_zero':f1_score(y,p,labels=list(range(8)),average='macro',zero_division=0),
                         'outside_reference_label_predictions':sum(v not in six for v in p),
                         'confusion_matrix_eight_classes':confusion_matrix(y,p,labels=list(range(8))).tolist(),
                         'per_class_f1':dict(zip(CLASSES,f1_score(y,p,labels=list(range(8)),average=None,zero_division=0).tolist())),
                         'mean_max_probability':float(np.mean([r['max_probability'] for r in rr])),
                         'wrong_mean_max_probability':float(np.mean([r['max_probability'] for r in rr if not r['correct']])) if any(not r['correct'] for r in rr) else None}
    summary={'plan_sha256':sha(OUT/'plan.json'),'completed_utc':datetime.now(timezone.utc).isoformat(),
             'planned':24,'successful':len(quality),'failed':failed,'class_order':CLASSES,'quality':quality,'metrics':metrics}
    save(OUT/'summary.json',clean(summary))
    print(json.dumps(clean(summary['metrics']),indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else run()
