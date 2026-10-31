"""Fetch pinned upstream-only WavLM model; no emotion-finetuned weights."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REVISION='4c66d4806a428f2e922ccfa1a962776e232d487b'
MODEL='microsoft/wavlm-base-plus'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assets=ROOT/'assets';assets.mkdir(exist_ok=True);registry=assets/'model_registry.json'
    if registry.exists():meta=json.loads(registry.read_text())
    else:
        meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true',timeout=120))
        registry.write_text(json.dumps(meta,indent=2))
    assert meta['sha']==REVISION
    out=assets/'wavlm-base-plus';out.mkdir(exist_ok=True)
    for name in ['config.json','preprocessor_config.json','README.md','pytorch_model.bin']:
        target=out/name
        expected=next((x.get('lfs',{}).get('sha256') for x in meta['siblings'] if x['rfilename']==name),None)
        if target.exists():
            if expected:assert sha(target)==expected,'Existing upstream weight checksum mismatch'
            continue
        url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true&ts={int(time.time())}'
        partial=target.with_suffix(target.suffix+'.partial');print('Downloading',name,flush=True)
        req=urllib.request.Request(url,headers={'User-Agent':'PRS-M2-Research/1.0'})
        with urllib.request.urlopen(req,timeout=120) as src,partial.open('wb') as dst:
            count=0
            while True:
                block=src.read(1024*1024)
                if not block:break
                dst.write(block);count+=len(block)
                if count%(32*1024*1024)==0:print(round(count/2**20),'MiB',flush=True)
        if expected:assert sha(partial)==expected,'Upstream checksum mismatch'
        partial.rename(target)
    print('Pinned WavLM model ready:',REVISION)
if __name__=='__main__':main()
