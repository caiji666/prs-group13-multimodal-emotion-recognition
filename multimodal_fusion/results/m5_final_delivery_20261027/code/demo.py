"""Synthetic, class-correlated features for integration tests, never research evidence."""
from pathlib import Path
import numpy as np
from . import CLASSES


def generate(directory, seed=42):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    # Full speech trial structure: 60 recordings per actor; neutral has one intensity.
    trials = [(a, e, intensity, s, r) for a in range(1, 25) for e in range(8)
              for intensity in ([1] if e == 0 else [1, 2]) for s in (1, 2) for r in (1, 2)]
    ids = np.array([f'01-{e+1:02d}-{i:02d}-{s:02d}-{r:02d}-{a:02d}' for a,e,i,s,r in trials])
    labels = np.array([t[1] for t in trials], dtype=np.int64)
    actors = np.array([t[0] for t in trials], dtype=np.int64)
    n = len(trials)
    # Different noisy class prototypes, actor nuisance and sample-specific modality corruption.
    for modality, dim, seq_dim, time, noise in [('audio', 32, 24, 12, 1.5), ('visual', 40, 28, 16, 1.7)]:
        prototype = rng.normal(size=(8, dim))
        seq_proto = rng.normal(size=(8, seq_dim))
        actor_shift = rng.normal(scale=.4, size=(24, dim))
        reliability = rng.choice([.7, 2.6], size=n, p=[.75, .25])
        pooled = prototype[labels] + actor_shift[actors-1] + rng.normal(size=(n, dim)) * reliability[:, None]
        logits = rng.normal(scale=noise, size=(n, 8))
        logits[np.arange(n), labels] += 2.5
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        lengths = rng.integers(time // 2, time + 1, size=n)
        sequence = seq_proto[labels, None, :] + rng.normal(scale=1.5, size=(n, time, seq_dim))
        sequence[np.arange(time)[None, :] >= lengths[:, None]] = 0
        # Deliberately shuffle visual rows to exercise ID-based pairing.
        order = rng.permutation(n) if modality == 'visual' else np.arange(n)
        np.savez_compressed(directory / f'{modality}.npz', data_kind=np.array('synthetic'), class_names=np.array(CLASSES),
                            sample_ids=ids[order], labels=labels[order], actors=actors[order],
                            pooled=pooled[order].astype('float32'), probs=probs[order].astype('float32'),
                            sequence=sequence[order].astype('float32'), lengths=lengths[order])
    return directory / 'audio.npz', directory / 'visual.npz'
