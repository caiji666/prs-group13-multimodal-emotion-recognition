"""Reproducible train-only augmented MFCC cache, shared across candidates/experiments."""
import hashlib
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from .data import extract

class EpochAugmentationCache:
    def __init__(self,rows,preprocessing,seed,root):
        if not rows or any(r['split']!='train' for r in rows):
            raise ValueError('Augmentation cache accepts training samples only')
        self.rows,self.preprocessing,self.seed=rows,preprocessing,seed
        key=dict(seed=seed,preprocessing=preprocessing,
                 extractor_sha256=hashlib.sha256(Path(__file__).with_name('data.py').read_bytes()).hexdigest(),
                 samples=[(r['sample_id'],str(Path(r['path']).resolve()),Path(r['path']).stat().st_size,Path(r['path']).stat().st_mtime_ns) for r in rows])
        digest=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        self.root=Path(root)/digest; self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'cache_metadata.json').write_text(json.dumps(key,indent=2))
        self.epoch=None; self.features=None
    def get(self,epoch,index):
        if self.epoch!=epoch:
            path=self.root/f'epoch_{epoch:03d}.pt'
            with (self.root/f'epoch_{epoch:03d}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if path.exists():
                    data=torch.load(path,map_location='cpu',weights_only=True)
                else:
                    data=[]
                    for i,row in enumerate(self.rows):
                        rng=np.random.default_rng(np.random.SeedSequence([self.seed,epoch,i]))
                        data.append(extract(row['path'],self.preprocessing,'train',augment=True,rng=rng))
                    temporary=path.with_suffix('.tmp')
                    torch.save(data,temporary); temporary.replace(path)
            if len(data)!=len(self.rows) or any(x.ndim!=2 or x.shape[1]!=self.preprocessing['n_mfcc'] or not torch.isfinite(x).all() for x in data):
                raise ValueError('Corrupted augmentation cache')
            self.epoch,self.features=epoch,data
        return self.features[index]
