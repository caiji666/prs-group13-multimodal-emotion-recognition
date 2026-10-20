"""Cache frozen upstream speech states for train/validation only."""
import json,hashlib,time
from pathlib import Path
import librosa,numpy as np,torch
from transformers import WavLMModel,Wav2Vec2FeatureExtractor
from ravdess_m2.data import manifest,LABELS
ROOT=Path(__file__).resolve().parent
REVISION='4c66d4806a428f2e922ccfa1a962776e232d487b'
LAYERS=[6,9,12]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wave(path):
    y,_=librosa.load(path,sr=16000,mono=True)
    y,_=librosa.effects.trim(y,top_db=35)
    if len(y)<400 or not np.isfinite(y).all():raise ValueError('Invalid waveform')
    return y

def main():
    torch.set_num_threads(4)
    data=ROOT.parents[1]/'data/raw';split=ROOT.parents[3]/'multimodal_fusion/configs/actor_splits.csv'
    rows=manifest(data,split);rows=[x for x in rows if x['split'] in ('train','val')]
    assert len(rows)==1200
    out=ROOT/'cache';out.mkdir(exist_ok=True)
    processor=Wav2Vec2FeatureExtractor.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True)
    model=WavLMModel.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True).eval().requires_grad_(False)
    features=[];start=time.monotonic()
    for i,row in enumerate(rows):
        dest=out/(row['sample_id']+'.pt')
        if dest.exists():
            item=torch.load(dest,map_location='cpu',weights_only=True)
            assert item['sample_id']==row['sample_id'] and item['audio_sha256']==sha(row['path'])
        else:
            inputs=processor(wave(row['path']),sampling_rate=16000,return_tensors='pt')
            with torch.inference_mode():
                result=model(**inputs,output_hidden_states=True)
            h=torch.stack([result.hidden_states[j][0] for j in LAYERS]).cpu().to(torch.float16)
            # Explicit float16 cache quantization, shared by extraction and inference.
            a=h.float();pooled=torch.cat([a.mean(1),a.std(1,unbiased=False)],dim=-1)
            item={k:row[k] for k in ('sample_id','actor_id','split','label')}
            item.update(hidden=h,pooled=pooled,audio_sha256=sha(row['path']))
            torch.save(item,dest)
        features.append({k:item[k] for k in ('sample_id','actor_id','split','label','pooled')})
        if (i+1)%50==0:print(f'Cached {i+1}/1200 train/val clips; elapsed {time.monotonic()-start:.1f}s',flush=True)
    payload={'rows':[{k:v for k,v in x.items() if k!='pooled'} for x in features],'pooled':torch.stack([x['pooled'] for x in features]),'layers':LAYERS,'model_revision':REVISION,'model_sha256':sha(ROOT/'assets/wavlm-base-plus/pytorch_model.bin'),'split_sha256':sha(split),'class_names':LABELS}
    torch.save(payload,out/'pooled_index.pt')
    (out/'metadata.json').write_text(json.dumps(dict(samples=1200,train=960,val=240,test_decoded=False,model_id='microsoft/wavlm-base-plus',revision=REVISION,model_sha256=payload['model_sha256'],layers=LAYERS,hidden_dim=768,cache_dtype='float16; pooled statistics computed after dequantization',preprocessing='mono16k, trim35dB, official do_normalize=false; no peak scaling; no augmentation during frozen extraction',elapsed_seconds=time.monotonic()-start),indent=2))
    print('Frozen cache completed; no test audio decoded.',flush=True)
if __name__=='__main__':main()
