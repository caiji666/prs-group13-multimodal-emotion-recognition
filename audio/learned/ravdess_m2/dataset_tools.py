import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .data import manifest, extract, collate, LABELS
from .models import AudioEncoder

def main():
    parser = argparse.ArgumentParser(description='Stage 1/2 data validation, MFCC extraction and checkpoint feature export')
    parser.add_argument('command', choices=['validate','extract','smoke','export'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--splits', default=str(Path(__file__).resolve().parents[3]/'multimodal_fusion/configs/actor_splits.csv'))
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output', default='artifacts')
    parser.add_argument('--encoder', choices=['cnn','bilstm','transformer'], default='cnn')
    parser.add_argument('--checkpoint', help='Required for export: state_dict with encoder, model_state_dict keys')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(config['seed']); np.random.seed(config['seed'])
    rows = manifest(args.data,args.splits,full=not args.allow_partial)
    out = Path(args.output); out.mkdir(parents=True,exist_ok=True)
    summary = dict(samples=len(rows), splits=dict(Counter(r['split'] for r in rows)), labels=LABELS, split_sha256=hashlib.sha256(Path(args.splits).read_bytes()).hexdigest(), config=config)
    # All audio is decoded and all MFCC sequences checked, even for validate.
    if args.command in ('validate','extract'):
        for row in rows:
            x = extract(row['path'],config,row['split'])
            if args.command == 'extract':
                target = out/'mfcc'/row['split']; target.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(target/(row['sample_id']+'.npz'), mfcc=x.numpy(), sample_id=row['sample_id'], split=row['split'], label=row['label'], actor_id=row['actor_id'])
        with (out/'manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    elif args.command == 'smoke':
        items = [dict(r,features=extract(r['path'],config,r['split'])) for r in [next(r for r in rows if r['split']==s) for s in ('train','val','test')]]
        batch=collate(items)
        summary['models']={}
        for kind in ('cnn','bilstm','transformer'):
            model=AudioEncoder(kind).eval()
            with torch.no_grad(): result=model(batch['x'],batch['lengths'])
            assert result['logits'].shape==(3,8)
            assert torch.isfinite(result['logits']).all()
            summary['models'][kind]={k:list(v.shape) for k,v in result.items()}
        summary['weights']='random initialization: shapes only, not useful learned features'
    else:
        if not args.checkpoint:
            parser.error('export requires a trained checkpoint; random features are not deliverables')
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        if checkpoint['encoder'] != args.encoder:
            parser.error('Checkpoint encoder mismatch')
        if checkpoint.get('preprocessing') != config:
            parser.error('Checkpoint preprocessing differs from --config; use the original preprocessing file')
        if checkpoint.get('class_names') != LABELS:
            parser.error('Checkpoint class order mismatch')
        if checkpoint.get('provenance', {}).get('split_sha256') != summary['split_sha256']:
            parser.error('Checkpoint actor split mismatch')
        if checkpoint.get('provenance', {}).get('diagnostic', False):
            parser.error('Diagnostic smoke checkpoints cannot be exported as learned deliverables')
        model=AudioEncoder(args.encoder, input_dim=config['n_mfcc']).eval(); model.load_state_dict(checkpoint['model_state_dict'])
        summary['checkpoint_sha256']=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
        summary['encoder']=args.encoder
        for row in rows:
            x=extract(row['path'],config,row['split'])
            with torch.no_grad(): result=model(x[None],torch.tensor([len(x)]))
            target=out/args.encoder/row['split']; target.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target/(row['sample_id']+'.npz'),sample_id=row['sample_id'],split=row['split'],label=row['label'],actor_id=row['actor_id'],sequence=result['sequence'][0].numpy(),pooled=result['pooled'][0].numpy(),length=int(result['lengths'][0]),padding_mask=result['padding_mask'][0].numpy(),logits=result['logits'][0].numpy(),probabilities=result['logits'].softmax(-1)[0].numpy(),class_names=np.array(LABELS))
    (out/(args.command+'_report.json')).write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
