"""Step 2: face frames -> 2D-DWT handcrafted features.

Proposal Sec. 3: 2D-DWT of each aligned face -> LL/LH/HL/HH subbands;
mean / variance / energy / entropy / edge information per subband;
concatenate per frame; temporal mean pooling over the 16 frames.

The proposal does not say over which area the statistics are computed.
Default (v2): crop the inner face (drops most background/hair), then split each
subband into a 4x4 grid and compute the 5 statistics per cell, so local
changes (mouth, brows, eyes) are not averaged away:
    4 subbands x 16 cells x 5 stats = 320 features per frame.
`--grid 1 --inner-crop 224` reproduces v1 (whole subband, 20 features per frame).

Output: <out> (npz) with
    sample_id  [N]          split [N]   label [N]
    F_sequence [N, 16, D]   per-frame features (before pooling)
    F_pooled   [N, D]       temporal mean pooling of F_sequence
    feature_names [D]
    frame_idx [N, 16]  frame_time_sec [N, 16]  video_fps [N]   (copied from the face files)

Usage:
    uv run python -m m3.dwt_features --faces-dir data/faces --out data/dwt_features.npz
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pywt
from tqdm import tqdm

from m3.common import (actor_from_sample_id, check_frame_timing, label_from_sample_id,
                       load_actor_splits)

WAVELET = "haar"
SUBBANDS = ["LL", "LH", "HL", "HH"]
STATS = ["mean", "var", "energy", "entropy", "edge"]
HIST_BINS = 64
N_FRAMES = 16
FRAME_SIZE = 224  # shared M3/M4 face frames; never resized or overwritten here
MIN_CELL = 3      # smallest grid cell side (3x3 Sobel)


def entropy(x):
    """Shannon entropy (bits) of the coefficient histogram."""
    hist, _ = np.histogram(x, bins=HIST_BINS)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def edge(x):
    """Mean Sobel gradient magnitude: how strong the edges/lines are."""
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx ** 2 + gy ** 2).mean())


def feature_names(grid):
    if grid == 1:
        return [f"{b}_{s}" for b in SUBBANDS for s in STATS]
    return [f"{b}_r{i}c{j}_{s}" for b in SUBBANDS
            for i in range(grid) for j in range(grid) for s in STATS]


def subband_size(inner_crop):
    """Side length of each one-level DWT subband of an inner_crop x inner_crop image."""
    return pywt.dwt_coeff_len(inner_crop, pywt.Wavelet(WAVELET).dec_len, mode="symmetric")


def validate_config(inner_crop, grid, frame_size=FRAME_SIZE):
    """Raise ValueError unless (inner_crop, grid) can be applied to a frame_size face.

    Cells may differ by one pixel when the subband side is not divisible by grid
    (v2: 90 / 4 -> 22 or 23), but every cell must be at least MIN_CELL pixels.
    """
    for name, v in (("inner_crop", inner_crop), ("grid", grid)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    if inner_crop > frame_size:
        raise ValueError(f"inner_crop={inner_crop} is larger than the {frame_size}x{frame_size} face")
    side = subband_size(inner_crop)
    if side // grid < MIN_CELL:
        raise ValueError(f"grid={grid} on a {side}x{side} subband (inner_crop={inner_crop}) gives "
                         f"cells smaller than {MIN_CELL}x{MIN_CELL}")


def frame_features(rgb, grid=4, inner_crop=180):
    if rgb.shape != (FRAME_SIZE, FRAME_SIZE, 3):
        raise ValueError(f"Expected a {FRAME_SIZE}x{FRAME_SIZE}x3 face frame, got {rgb.shape}")
    validate_config(inner_crop, grid)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    m = (gray.shape[0] - inner_crop) // 2
    gray = gray[m:m + inner_crop, m:m + inner_crop]
    LL, (LH, HL, HH) = pywt.dwt2(gray, WAVELET)
    feats = []
    for band in (LL, LH, HL, HH):
        band = band.astype(np.float32)
        h, w = band.shape
        for i in range(grid):
            for j in range(grid):
                cell = band[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
                feats += [cell.mean(), cell.var(), (cell ** 2).mean(), entropy(cell), edge(cell)]
    return np.array(feats, dtype=np.float32)


def clip_features(frames, grid=4, inner_crop=180):
    """[16, 224, 224, 3] face frames -> [16, D] per-frame features (input is not modified)."""
    return np.stack([frame_features(fr, grid, inner_crop) for fr in frames])


def face_files(faces_dir):
    """Face npz files sorted by sample_id (the order every output uses)."""
    return sorted(p for p in Path(faces_dir).glob("*.npz") if not p.stem.endswith(".tmp"))


def load_face_file(f):
    """Read one face file (read-only) after checking its timing metadata."""
    z = np.load(f)
    missing = [k for k in ("frame_idx", "frame_time_sec", "video_fps") if k not in z.files]
    if missing:
        raise KeyError(f"{f.name} has no {missing}; run `python -m m3.add_frame_timestamps` first")
    check_frame_timing(z["frame_idx"], z["frame_time_sec"], z["video_fps"])
    return z


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces-dir", default="data/faces")
    ap.add_argument("--out", default="data/dwt_features.npz")
    ap.add_argument("--grid", type=int, default=4, help="cells per side (1 = whole subband, v1)")
    ap.add_argument("--inner-crop", type=int, default=180,
                    help="centre crop of the 224x224 face before the DWT (224 = no crop, v1)")
    args = ap.parse_args(argv)
    validate_config(args.inner_crop, args.grid)

    splits = load_actor_splits()
    ids, seqs, idx, times, fps = [], [], [], [], []
    for f in tqdm(face_files(args.faces_dir)):
        z = load_face_file(f)
        seqs.append(clip_features(z["frames"], args.grid, args.inner_crop))
        idx.append(z["frame_idx"])
        times.append(z["frame_time_sec"])
        fps.append(z["video_fps"])
        ids.append(f.stem)

    F_sequence = np.stack(seqs)
    np.savez(
        args.out,
        sample_id=np.array(ids),
        split=np.array([splits[actor_from_sample_id(s)] for s in ids]),
        label=np.array([label_from_sample_id(s) for s in ids]),
        F_sequence=F_sequence,
        F_pooled=F_sequence.mean(axis=1),
        feature_names=np.array(feature_names(args.grid)),
        frame_idx=np.stack(idx),
        frame_time_sec=np.stack(times).astype(np.float32),
        video_fps=np.array(fps, dtype=np.float32),
    )
    print(f"Saved {len(ids)} samples, F_sequence {F_sequence.shape} -> {args.out}")


if __name__ == "__main__":
    main()
