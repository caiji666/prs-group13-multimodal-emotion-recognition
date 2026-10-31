import csv
from pathlib import Path
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def parse_filename(path):
    fields = Path(path).stem.split('-')
    if len(fields) != 7 or any(len(f) != 2 or not f.isdigit() for f in fields):
        raise ValueError(f'Invalid RAVDESS name: {path}')
    m, v, e, i, s, r, a = map(int, fields)
    if not (m == 3 and v == 1 and 1 <= e <= 8 and i in (1,2) and s in (1,2) and r in (1,2) and 1 <= a <= 24):
        raise ValueError(f'Expected audio-only speech: {path}')
    if e == 1 and i != 1:
        raise ValueError('Neutral has no strong intensity')
    return dict(sample_id='-'.join(fields[1:]), actor_id=a, label=e-1, emotion=LABELS[e-1])

def read_splits(path):
    # No fallback or random split is permitted.
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {'actor_id', 'split'}:
            raise ValueError('CSV must have actor_id,split headers; explicitly adapt original schema if needed')
        result = {}
        for row in reader:
            actor = int(row['actor_id'])
            split = row['split'].strip()
            if actor in result or actor not in range(1,25) or split not in ('train','val','test'):
                raise ValueError(f'Duplicate/invalid actor or split: {row}')
            result[actor] = split
    if set(result) != set(range(1,25)) or set(result.values()) != {'train','val','test'}:
        raise ValueError('Require all 24 actors, each exactly once, and all three splits')
    return result

def manifest(root, splits, full=True):
    assignment = read_splits(splits)
    rows = []
    seen = set()
    for path in sorted(Path(root).rglob('*.wav')):
        # Song or other modalities must not enter the speech pipeline.
        fields = path.stem.split('-')
        if len(fields) == 7 and fields[:2] != ['03','01']:
            continue
        row = parse_filename(path)
        if row['sample_id'] in seen:
            raise ValueError(f'Duplicate sample_id: {path}')
        seen.add(row['sample_id'])
        rows.append(dict(row, path=str(path), split=assignment[row['actor_id']]))
    if not rows:
        raise ValueError('No speech WAV files found')
    if full:
        from collections import Counter
        if len(rows) != 1440 or Counter(r['actor_id'] for r in rows) != {a:60 for a in range(1,25)}:
            raise ValueError('Official full speech subset must contain 1440 samples, 60 per actor')
    return rows

def augment_wave(y, sr, rng):
    # Piecewise monotonic time warping, retaining waveform duration.
    knots = np.linspace(0, len(y)-1, 6)
    rates = rng.uniform(.85, 1.15, 5)
    warped = np.r_[0, np.cumsum(rates)]
    warped = warped / warped[-1] * (len(y)-1)
    y = np.interp(np.interp(np.arange(len(y)), knots, warped), np.arange(len(y)), y)
    y = y * rng.uniform(.8, 1.1)
    rms = max(float(np.sqrt(np.mean(y*y))), 1e-8)
    y += rng.normal(0, rms * 10 ** (-rng.uniform(20,35)/20), len(y))
    return np.clip(y, -1, 1).astype(np.float32)

def extract(path, config, split, augment=False, rng=None):
    if split not in ('train','val','test'):
        raise ValueError('Invalid split')
    if augment and split != 'train':
        raise ValueError('Augmentation is allowed only for training')
    sr = config['sample_rate']
    y, _ = librosa.load(path, sr=sr, mono=True)
    if not len(y) or not np.isfinite(y).all():
        raise ValueError(f'Empty or non-finite audio: {path}')
    y, _ = librosa.effects.trim(y, top_db=config['trim_db'])
    if np.max(np.abs(y), initial=0) < 1e-8:
        raise ValueError(f'Silent audio: {path}')
    y = y / np.max(np.abs(y)) * .95
    if config.get('denoise', False):
        # Optional stationary spectral gate, using low-energy frames as noise estimate.
        z = librosa.stft(y, n_fft=config['n_fft'], hop_length=config['hop_length'])
        mag = np.abs(z)
        noise = np.quantile(mag, .1, axis=1, keepdims=True)
        z *= np.maximum(0.1, 1-noise / np.maximum(mag,1e-8))
        y = librosa.istft(z, hop_length=config['hop_length'], length=len(y))
    if augment:
        y = augment_wave(y, sr, rng or np.random.default_rng())
    y = np.pad(y, (0,max(0,config['n_fft']-len(y))))
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config['n_mfcc'], n_fft=config['n_fft'], hop_length=config['hop_length'], n_mels=config['n_mels']).T.copy()
    if not np.isfinite(features).all():
        raise ValueError('Non-finite MFCC')
    return torch.from_numpy(features)

class SpeechDataset(Dataset):
    def __init__(self, rows, config, split, augment=False):
        if augment and split != 'train':
            raise ValueError('Augmentation is allowed only for training')
        self.rows = [r for r in rows if r['split'] == split]
        self.config, self.split, self.augment = config, split, augment
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        # torch worker seeds also cover NumPy through this seed draw.
        rng = np.random.default_rng(torch.randint(0,2**31-1,()).item())
        return dict(row, features=extract(row['path'], self.config, self.split, self.augment, rng))

def collate(items):
    lengths = torch.tensor([len(i['features']) for i in items])
    x = torch.nn.utils.rnn.pad_sequence([i['features'] for i in items], batch_first=True)
    mask = torch.arange(x.shape[1])[None,:] >= lengths[:,None]
    return dict(x=x, lengths=lengths, padding_mask=mask, labels=torch.tensor([i['label'] for i in items]), sample_ids=[i['sample_id'] for i in items], splits=[i['split'] for i in items])
