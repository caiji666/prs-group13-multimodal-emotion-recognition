"""Exercise the running local UI backend. Synthetic media test, not emotion evaluation."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

import numpy as np
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent
BASE='http://127.0.0.1:8765'

def request(path,data=None,headers=None):
    req=urllib.request.Request(BASE+path,data=data,headers=headers or {})
    return urllib.request.urlopen(req,timeout=20)

def job(path,payload,headers=None):
    with request(path,payload,headers) as r:token=json.load(r)['job_id']
    for _ in range(240):
        with request('/api/jobs/'+token) as r:state=json.load(r)
        if state['status'] in ('completed','failed'):return state
        time.sleep(.25)
    raise TimeoutError('Inference job did not complete within 60 seconds.')

def main():
    out=ROOT/'test-artifacts';out.mkdir(exist_ok=True)
    video=out/'engineering_smoke.mp4'
    if not video.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','lavfi','-i','color=c=steelblue:s=320x240:r=12:d=2','-f','lavfi','-i','sine=frequency=440:sample_rate=16000:duration=2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-shortest','-y',str(video)],check=True,timeout=30)
    checks={}
    with request('/api/catalog') as r:catalog=json.load(r)
    assert catalog['raw_video']['ready'] and len(catalog['samples'])==240
    checks['catalog_and_raw_readiness']=True
    for method in ('gated','cross_attention','decision_tuned'):
        payload=json.dumps({'sample_id':catalog['samples'][0],'audio':'learned','visual':'hc','method':method}).encode()
        state=job('/api/replay',payload,{'Content-Type':'application/json'})
        assert state['status']=='completed',state
        assert state['result']['input']['source_kind']=='validation_replay'
        checks['replay_'+method]=state['id']
    content=video.read_bytes()
    for audio,method in [('learned','cross_attention'),('combined','audio_only')]:
        state=job(f'/api/upload?audio={audio}&visual=hc&method={method}&face_mode=mediapipe',content,
                  {'Content-Type':'application/octet-stream','X-File-Name':'01-01-01-01-01-09.mp4'})
        assert state['status']=='completed',state
        r=state['result'];assert r['input']['source_kind']=='uploaded_video'
        assert r['input']['sample_id']=='external_'+hashlib.sha256(content).hexdigest()[:20]
        assert r['input']['ground_truth'] is None
        with request('/api/jobs/'+state['id']+'/files/extracted_features.npz') as response:
            with np.load(io.BytesIO(response.read()),allow_pickle=False) as z:
                assert z['audio_pooled'].shape==(1,128 if audio=='learned' else 176)
                assert z['visual_sequence'].shape==(1,16,320)
                assert z['sample_ids'][0]==r['input']['sample_id']
        with request('/api/jobs/'+state['id']+'/files/audio.wav',headers={'Range':'bytes=0-9'}) as response:
            assert response.status==206 and len(response.read())==10
        checks['raw_'+audio+'_'+method]=state['id']
    # Origin checks and invalid representation combinations fail before inference.
    for label,path,body,headers in [
        ('cross_origin','/api/catalog',None,{'Origin':'http://example.invalid'}),
        ('unknown_path','/../demo/server.py',None,{}),
        ('incompatible_raw_features','/api/upload?visual=deep',b'not video',{'X-File-Name':'video.mp4'}),
        ('test_replay','/api/replay',json.dumps({'sample_id':'01-01-01-01-01-02','audio':'learned','visual':'hc','method':'concat'}).encode(),{'Content-Type':'application/json'}),
    ]:
        try:request(path,body,headers)
        except urllib.error.HTTPError as error:assert error.code in (400,404);checks[label+'_rejected']=True
        else:raise AssertionError('Expected rejection: '+label)
    no_audio=out/'no_audio.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(video),'-an','-c:v','copy','-y',str(no_audio)],check=True,timeout=30)
    bad=job('/api/upload',no_audio.read_bytes(),{'X-File-Name':'no_audio.mp4','Content-Type':'application/octet-stream'})
    assert bad['status']=='failed' and 'audio track' in bad['error'];checks['missing_audio_reports_error']=True
    report={'scope':'synthetic AV engineering smoke only; no external emotion accuracy claim','checks':checks,'passed':True}
    (out/'http_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
