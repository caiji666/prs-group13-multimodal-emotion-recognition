"""Strict pairing by trial ID, actor partitioning and train-only normalization."""
import json
import re
from pathlib import Path
import numpy as np
from . import CLASSES


def trial_id(filename):
    """Drop only modality; retain vocal channel/emotion/intensity/statement/repetition/actor."""
    parts = Path(filename).stem.split('-')
    if len(parts) != 7 or any(not re.fullmatch(r'\d{2}', p) for p in parts):
        raise ValueError(f'Invalid RAVDESS filename: {filename}')
    modality, channel, emotion, intensity, statement, repetition, actor = map(int, parts)
    if modality not in (1, 2, 3) or channel not in (1, 2):
        raise ValueError(f'Invalid modality/channel: {filename}')
    if not (1 <= emotion <= 8 and intensity in (1, 2) and statement in (1, 2)
            and repetition in (1, 2) and 1 <= actor <= 24):
        raise ValueError(f'Invalid trial fields: {filename}')
    if channel != 1:
        raise ValueError(f'Song excluded: {filename}')
    if emotion == 1 and intensity != 1:
        raise ValueError(f'Neutral has normal intensity only: {filename}')
    return '-'.join(parts[1:])


def load_modality(path):
    with np.load(path, allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    required = {'sample_ids', 'labels', 'actors', 'class_names', 'pooled', 'probs'}
    if required - d.keys():
        raise ValueError(f'{path}: missing {sorted(required - d.keys())}')
    if tuple(d['class_names'].tolist()) != CLASSES:
        raise ValueError('class_names must follow the documented eight-class order')
    ids = d['sample_ids']
    n = len(ids)
    if n == 0 or ids.ndim != 1 or ids.dtype.kind not in 'US' or len(set(ids.tolist())) != n:
        raise ValueError('sample_ids must be unique nonempty strings')
    for k in ('labels', 'actors'):
        if d[k].shape != (n,) or d[k].dtype.kind not in 'iu':
            raise ValueError(f'{k} must be integer [N]')
    for sid, label, actor in zip(ids, d['labels'], d['actors']):
        if trial_id('03-' + str(sid) + '.wav') != sid:
            raise ValueError(f'Noncanonical sample ID {sid}')
        fields = list(map(int, str(sid).split('-')))
        if label != fields[1] - 1 or actor != fields[-1]:
            raise ValueError(f'Label/actor inconsistent with sample ID {sid}')
    if d['pooled'].ndim != 2 or d['pooled'].shape[0] != n or d['pooled'].shape[1] < 1:
        raise ValueError('pooled must be [N,D]')
    p = d['probs']
    if p.shape != (n, 8) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError('probs must be finite nonnegative [N,8] probabilities summing to one')
    if not np.isfinite(d['pooled']).all():
        raise ValueError('Nonfinite pooled features')
    if ('sequence' in d) != ('lengths' in d):
        raise ValueError('Provide sequence and lengths together')
    if 'sequence' in d:
        s, lengths = d['sequence'], d['lengths']
        if s.ndim != 3 or s.shape[0] != n or s.shape[2] < 1 or not np.isfinite(s).all():
            raise ValueError('sequence must be finite [N,T,D]')
        if lengths.shape != (n,) or lengths.dtype.kind not in 'iu' or (lengths < 1).any() or (lengths > s.shape[1]).any():
            raise ValueError('lengths must be integer [N] in 1..T')
    return d


def load_pair(audio_path, visual_path):
    a, v = load_modality(audio_path), load_modality(visual_path)
    if set(a['sample_ids'].tolist()) != set(v['sample_ids'].tolist()):
        raise ValueError('Modalities have different sample IDs; missing pairs must be resolved explicitly')
    lookup = {sid: i for i, sid in enumerate(v['sample_ids'])}
    order = np.array([lookup[sid] for sid in a['sample_ids']])
    v = {k: x[order] if k != 'class_names' and x.ndim and x.shape[0] == len(order) else x for k, x in v.items()}
    if not np.array_equal(a['labels'], v['labels']) or not np.array_equal(a['actors'], v['actors']):
        raise ValueError('Audio and visual labels/actors disagree')
    d = {k: a[k] for k in ('sample_ids', 'labels', 'actors')}
    kinds = {str(source.get('data_kind', 'unspecified')) for source in (a, v)}
    if len(kinds) != 1:
        raise ValueError('Audio and visual data_kind disagree')
    d['data_kind'] = kinds.pop()
    for prefix, source in (('audio', a), ('visual', v)):
        for k in ('pooled', 'probs', 'sequence', 'lengths'):
            if k in source:
                d[prefix + '_' + k] = source[k].astype(np.int64 if k == 'lengths' else np.float32)
    return d


def split_indices(d, config_path):
    split = json.loads(Path(config_path).read_text())
    if set(split) != {'train', 'val', 'test'}:
        raise ValueError('Partition must contain train, val, test')
    seen = set()
    for name, actors in split.items():
        if not actors or any(type(x) is not int or not 1 <= x <= 24 for x in actors):
            raise ValueError(f'Invalid actors in {name}')
        if len(actors) != len(set(actors)) or seen.intersection(actors):
            raise ValueError('Actor leakage/duplicate in partition')
        seen.update(actors)
    if set(d['actors'].tolist()) - seen:
        raise ValueError('Some samples have no partition')
    indices = {k: np.flatnonzero(np.isin(d['actors'], v)) for k, v in split.items()}
    for name, ix in indices.items():
        if not len(ix) or set(d['labels'][ix].tolist()) != set(range(8)):
            raise ValueError(f'{name} must contain samples of all eight classes')
    return indices, split


def fit_normalizers(d, train):
    stats = {}
    for k, x in d.items():
        if k.endswith('_pooled') or k.endswith('_sequence'):
            x = x[train]
            if k.endswith('_sequence'):
                lengths = d[k.replace('sequence', 'lengths')][train]
                x = x[np.arange(x.shape[1])[None, :] < lengths[:, None]]
            stats[k] = {'mean': x.mean(0).tolist(), 'std': np.maximum(x.std(0), 1e-6).tolist()}
    return stats


def normalize(d, stats):
    out = dict(d)
    for k, stat in stats.items():
        x = (d[k] - np.asarray(stat['mean'], np.float32)) / np.asarray(stat['std'], np.float32)
        if k.endswith('_sequence'):
            lengths = d[k.replace('sequence', 'lengths')]
            x = np.where((np.arange(x.shape[1])[None, :] < lengths[:, None])[..., None], x, 0)
        out[k] = x.astype(np.float32)
    return out
