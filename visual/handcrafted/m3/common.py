"""Team-wide conventions: sample_id, split, label (see RAVDESS_Sample_ID_Convention.md)."""
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Class order = RAVDESS emotion code order (01..08 -> 0..7).
# Confirm with the team that this matches Yurun's class order before sending.
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def generate_sample_id(filename):
    """Copied unchanged from the team convention file."""
    parts = Path(filename).stem.split("-")
    if len(parts) != 7 or any(
        len(part) != 2 or not part.isdigit() for part in parts
    ):
        raise ValueError("Expected a RAVDESS filename with seven two-digit fields.")
    if parts[1] != "01":
        raise ValueError("This project convention applies to speech samples only.")
    return "-".join(parts[1:])


def label_from_sample_id(sample_id):
    """Emotion is the 2nd field of the sample_id; code 01..08 -> label 0..7."""
    return int(sample_id.split("-")[1]) - 1


def actor_from_sample_id(sample_id):
    return int(sample_id.split("-")[-1])


def load_actor_splits(path=ROOT / "actor_splits.csv"):
    df = pd.read_csv(path)
    return dict(zip(df["actor_id"].astype(int), df["split"]))


def read_video_fps(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def frame_times(frame_idx, fps):
    """Time (s) of each frame in the original video: frame_idx / fps, as float32.

    Raises ValueError on a non-finite / non-positive fps or a negative /
    decreasing frame_idx, so wrong timestamps are never written silently.
    """
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}")
    idx = np.asarray(frame_idx)
    if idx.ndim != 1 or (idx < 0).any() or (np.diff(idx) < 0).any():
        raise ValueError(f"Invalid frame_idx (must be 1-D, >= 0, non-decreasing): {idx}")
    return (idx / fps).astype(np.float32)


def check_frame_timing(frame_idx, frame_time_sec, video_fps, n_frames=16):
    """Validate stored timing metadata of one sample."""
    if np.shape(frame_idx) != (n_frames,) or np.shape(frame_time_sec) != (n_frames,):
        raise ValueError(f"Expected {n_frames} frame_idx / frame_time_sec values")
    expected = frame_times(frame_idx, video_fps)
    t = np.asarray(frame_time_sec)
    if (np.diff(t) < 0).any():
        raise ValueError("frame_time_sec is decreasing")
    if not np.allclose(t, expected, rtol=1e-6, atol=1e-6):
        raise ValueError("frame_time_sec does not match frame_idx / video_fps")


def find_videos(video_dir, modality="01"):
    """All speech videos of one modality (01 = audio-video, 02 = video-only).

    Both modalities give the same sample_id, so only one may be used.
    """
    videos = {}
    for p in sorted(Path(video_dir).rglob("*.mp4")):
        parts = p.stem.split("-")
        if len(parts) != 7 or parts[0] != modality or parts[1] != "01":
            continue
        sid = generate_sample_id(p.name)
        if sid in videos:
            raise ValueError(f"Duplicate sample_id {sid}: {videos[sid]} and {p}")
        videos[sid] = p
    return videos
