"""Backfill video_fps and frame_time_sec into existing face npz files.

The face files from step 1 were made before timing metadata was saved. This
reads each original video's FPS and computes frame_time_sec = frame_idx / fps
from the frame_idx already stored. It does NOT re-run MediaPipe, re-sample
frames or touch frames / face_found / frame_idx. Safe to re-run: files whose
timing fields are already correct are verified and skipped.

Usage:
    uv run python -m m3.add_frame_timestamps --video-dir "D:/RAVDESS/Video_Speech" --faces-dir data/faces --modality 01
"""
import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from m3.common import find_videos, frame_times, read_video_fps

N_FRAMES = 16
TIMING_KEYS = ("frame_time_sec", "video_fps")


def process_file(face_path, video_path):
    """Return 'updated' or 'skipped'; raise on any problem."""
    with np.load(face_path, allow_pickle=False) as z:
        original = {k: z[k] for k in z.files}
    for k in ("frames", "face_found", "frame_idx"):
        if k not in original:
            raise KeyError(f"missing field {k}")
    if original["frame_idx"].shape != (N_FRAMES,):
        raise ValueError(f"frame_idx has shape {original['frame_idx'].shape}, expected ({N_FRAMES},)")

    fps = np.float32(read_video_fps(video_path))
    t = frame_times(original["frame_idx"], fps)  # validates fps and frame_idx

    if all(k in original for k in TIMING_KEYS) \
            and original["video_fps"] == fps and np.array_equal(original["frame_time_sec"], t):
        return "skipped"

    keep = {k: v for k, v in original.items() if k not in TIMING_KEYS}
    tmp = face_path.with_name(face_path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **keep, frame_time_sec=t, video_fps=fps)
    with np.load(tmp, allow_pickle=False) as z:  # verify before replacing the original
        for k, v in keep.items():
            if z[k].dtype != v.dtype or not np.array_equal(z[k], v):
                tmp.unlink()
                raise RuntimeError(f"field {k} changed while rewriting")
    os.replace(tmp, face_path)
    return "updated"


def run(video_dir, faces_dir, modality="01", expected=1440):
    videos = find_videos(video_dir, modality)  # raises on duplicate sample_ids
    files = sorted(p for p in Path(faces_dir).glob("*.npz") if not p.stem.endswith(".tmp"))
    if len(files) != expected:
        raise ValueError(f"Found {len(files)} face files in {faces_dir}, expected {expected}")

    counts = {"updated": 0, "skipped": 0, "failed": 0}
    failures = []
    for f in tqdm(files):
        try:
            if f.stem not in videos:
                raise FileNotFoundError(f"no modality-{modality} video for sample_id {f.stem}")
            counts[process_file(f, videos[f.stem])] += 1
        except Exception as e:  # report every failure, then stop with an error
            counts["failed"] += 1
            failures.append(f"{f.name}: {e}")

    print(f"updated={counts['updated']} skipped={counts['skipped']} failed={counts['failed']}")
    if failures:
        print("\n".join(failures))
        raise SystemExit(f"{len(failures)} file(s) failed")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--faces-dir", default="data/faces")
    ap.add_argument("--modality", default="01", choices=["01", "02"])
    ap.add_argument("--expected", type=int, default=1440, help="number of face files expected")
    args = ap.parse_args(argv)
    run(args.video_dir, args.faces_dir, args.modality, args.expected)


if __name__ == "__main__":
    main()
