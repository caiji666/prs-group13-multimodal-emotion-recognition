"""NumPy-only Stage 4 reader: copy this file as handoff_reader.py for M1/M5."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def load_pooled(root, split):
    if split not in ('train','val','test'):
        raise ValueError('Expected train, val or test')
    with np.load(Path(root)/'pooled'/(split+'.npz'),allow_pickle=False) as z:
        result={key:z[key].copy() for key in z.files}
    if len(set(result['sample_ids'].tolist())) != len(result['sample_ids']):
        raise ValueError('Duplicate pooled sample IDs')
    return result


def align_pooled(root, split, sample_ids, labels, splits):
    """Return learned vectors in a partner table's exact order; never concatenate here."""
    data=load_pooled(root,split)
    ids=list(sample_ids)
    if len(set(ids))!=len(ids) or len(ids)!=len(labels) or len(ids)!=len(splits):
        raise ValueError('Duplicate IDs or inconsistent partner metadata lengths')
    lookup={sid:i for i,sid in enumerate(data['sample_ids'].tolist())}
    if any(sid not in lookup for sid in ids):
        raise ValueError('Partner sample IDs missing from selected split')
    indices=np.array([lookup[sid] for sid in ids],dtype=np.int64)
    if not np.array_equal(data['labels'][indices],np.asarray(labels)) or not np.array_equal(data['splits'][indices],np.asarray(splits)):
        raise ValueError('Partner labels/splits do not match')
    return data['pooled'][indices]


def load_sequence(root, sample_id):
    root=Path(root)
    # Resolve through the manifest; do not infer split from the ID or caller order.
    with (root/'manifest.csv').open(newline='') as f:
        matches=[r for r in csv.DictReader(f) if r['sample_id']==sample_id]
    if len(matches)!=1:
        raise ValueError('Sample ID missing or duplicated in manifest')
    path=(root/matches[0]['feature_file']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Unsafe feature path')
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key].copy() for key in z.files}


def pad_sequences(records):
    """Return [B,Tmax,d], lengths and mask. True means padding/ignored attention."""
    if not records:
        raise ValueError('Empty sequence batch')
    lengths=np.array([int(r['length']) for r in records],dtype=np.int64)
    if np.any(lengths<1):
        raise ValueError('Sequence lengths must be positive')
    dim=records[0]['sequence'].shape[1]
    x=np.zeros((len(records),int(lengths.max()),dim),dtype=np.float32)
    for i,(record,n) in enumerate(zip(records,lengths)):
        if record['sequence'].shape!=(n,dim):
            raise ValueError('Sequence shape mismatch')
        x[i,:n]=record['sequence']
    mask=np.arange(x.shape[1])[None,:]>=lengths[:,None]
    return dict(sequence=x,lengths=lengths,padding_mask=mask,
                sample_ids=np.array([str(r['sample_id']) for r in records]),
                labels=np.array([int(r['label']) for r in records]),
                splits=np.array([str(r['split']) for r in records]))


def validate_bundle(root):
    """Check every feature and its split/label/ID, without computing task accuracy."""
    root=Path(root)
    metadata=json.loads((root/'metadata.json').read_text())
    with (root/'manifest.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    with (root/'actor_splits.csv').open(newline='') as f:
        assignment={int(r['actor_id']):r['split'] for r in csv.DictReader(f)}
    if set(assignment)!=set(range(1,25)):
        raise ValueError('Missing actor assignment')
    if len(rows)!=metadata['samples'] or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Manifest count/ID uniqueness mismatch')
    expected_dim=metadata['feature_dim']
    counts={s:0 for s in ('train','val','test')}
    lengths=[]
    aggregates={s:load_pooled(root,s) for s in counts}
    lookup={s:{sid:i for i,sid in enumerate(z['sample_ids'].tolist())} for s,z in aggregates.items()}
    for row in rows:
        sid,split,label,actor=row['sample_id'],row['split'],int(row['label']),int(row['actor_id'])
        fields=sid.split('-')
        if len(fields)!=6 or fields[0]!='01' or int(fields[-1])!=actor or int(fields[1])-1!=label or assignment[actor]!=split:
            raise ValueError('Invalid sample ID, label or actor split')
        path=(root/row['feature_file']).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Feature file path/hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            n=int(z['length']); sequence=z['sequence']; pooled=z['pooled']
            if str(z['sample_id'])!=sid or str(z['split'])!=split or int(z['label'])!=label or int(z['actor_id'])!=actor:
                raise ValueError('Per-sample metadata mismatch')
            if sequence.shape!=(n,expected_dim) or pooled.shape!=(expected_dim,) or n<1 or n!=int(row['sequence_length']):
                raise ValueError('Feature shape mismatch')
            if sequence.dtype!=np.float32 or pooled.dtype!=np.float32 or not np.isfinite(sequence).all() or not np.isfinite(pooled).all():
                raise ValueError('Nonfinite or non-float32 features')
            if z['padding_mask'].dtype!=np.bool_ or z['padding_mask'].shape!=(n,) or z['padding_mask'].any():
                raise ValueError('Per-sample export must be unpadded')
            np.testing.assert_allclose(sequence.mean(0),pooled,rtol=2e-5,atol=2e-6)
            aggregate=aggregates[split]; i=lookup[split][sid]
            if int(aggregate['labels'][i])!=label or str(aggregate['splits'][i])!=split or int(aggregate['actor_ids'][i])!=actor:
                raise ValueError('Pooled aggregate metadata mismatch')
            np.testing.assert_array_equal(aggregate['pooled'][i],pooled)
        counts[split]+=1; lengths.append(n)
    if counts!=metadata['split_counts'] or any(len(aggregates[s]['sample_ids'])!=counts[s] for s in counts):
        raise ValueError('Split counts mismatch')
    for split in counts:
        with (root/'pooled'/(split+'.csv')).open(newline='') as f:
            csv_rows=list(csv.DictReader(f))
        z=aggregates[split]
        if [r['sample_id'] for r in csv_rows]!=z['sample_ids'].tolist():
            raise ValueError('CSV/NPZ ID alignment mismatch')
        values=np.array([[float(r[f'f_{i:03d}']) for i in range(expected_dim)] for r in csv_rows],dtype=np.float32)
        np.testing.assert_allclose(values,z['pooled'],rtol=1e-6,atol=1e-7)
    checkpoint=root/'checkpoint/best.pt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=metadata['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    if hashlib.sha256((root/'actor_splits.csv').read_bytes()).hexdigest()!=metadata['split_sha256']:
        raise ValueError('Actor split hash mismatch')
    return dict(valid=True,samples=len(rows),split_counts=counts,feature_dim=expected_dim,
                sequence_length_min=min(lengths),sequence_length_max=max(lengths),
                sequence_length_mean=float(np.mean(lengths)),test_metrics_computed=False)
