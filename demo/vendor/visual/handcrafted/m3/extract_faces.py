"""Step 1: video -> 16 aligned 224x224 face frames per sample.

Proposal Sec. 3: uniformly sample 16 frames, MediaPipe face/landmark detection,
OpenCV alignment + crop, resize to 224x224.

Output: <out_dir>/<sample_id>.npz with
    frames         uint8   [16, 224, 224, 3]  (RGB)
    face_found     bool    [16]               (False = box borrowed from a neighbour frame)
    frame_idx      int     [16]               (original frame numbers)
    frame_time_sec float32 [16]               (frame_idx / video_fps, seconds)
    video_fps      float32 scalar             (FPS reported by OpenCV)
plus <out_dir>/face_log.csv listing frames where no face was detected.

Usage:
    uv run python -m m3.extract_faces --video-dir D:/RAVDESS --out-dir data/faces
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from m3.common import find_videos, frame_times

N_FRAMES = 16
SIZE = 224
MARGIN = 1.3  # enlarge the detected box so the whole face (forehead, chin) is kept

_detector = None


def _init_worker():
    global _detector
    import mediapipe as mp
    _detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    )


def read_frames(path, n=N_FRAMES):
    """Uniformly sample n frames over the whole clip (RGB). Returns frames, frame_idx, fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = np.linspace(0, max(total - 1, 0), n).round().astype(int)
    frames, idx, last = [], 0, None
    for target in wanted:
        while idx <= target:
            ok, frame = cap.read()
            if not ok:
                break
            last = frame
            idx += 1
        if last is None:
            raise RuntimeError(f"Cannot read video {path}")
        frames.append(cv2.cvtColor(last, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, wanted, fps


def detect(frame):
    """Return (box cx, cy, side, eye angle in degrees) or None."""
    h, w = frame.shape[:2]
    res = _detector.process(frame)
    if not res.detections:
        return None
    det = max(res.detections, key=lambda d: d.score[0])
    bb = det.location_data.relative_bounding_box
    kp = det.location_data.relative_keypoints  # 0 = right eye, 1 = left eye
    cx = (bb.xmin + bb.width / 2) * w
    cy = (bb.ymin + bb.height / 2) * h
    side = max(bb.width * w, bb.height * h) * MARGIN
    dx = (kp[1].x - kp[0].x) * w
    dy = (kp[1].y - kp[0].y) * h
    angle = np.degrees(np.arctan2(dy, dx))
    return cx, cy, side, angle


def align_crop(frame, box):
    """Rotate so the eyes are horizontal, then crop a square around the face."""
    cx, cy, side, angle = box
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             borderMode=cv2.BORDER_REPLICATE)
    half = side / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))
    pad = max(0, -x0, -y0, x1 - frame.shape[1], y1 - frame.shape[0])
    if pad:
        rotated = cv2.copyMakeBorder(rotated, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
    return cv2.resize(rotated[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def process(args):
    sample_id, path, out_path = args
    frames, frame_idx, fps = read_frames(path)
    frame_time_sec = frame_times(frame_idx, fps)  # raises on an invalid FPS
    boxes = [detect(f) for f in frames]
    found = np.array([b is not None for b in boxes])
    if not found.any():
        # No face anywhere: fall back to a centre crop so the sample is not lost.
        h, w = frames[0].shape[:2]
        boxes = [(w / 2, h / 2, min(h, w), 0.0)] * len(frames)
    else:
        # Missing frames borrow the box of the nearest frame where a face was found.
        good = np.flatnonzero(found)
        boxes = [b if b is not None else boxes[good[np.abs(good - i).argmin()]]
                 for i, b in enumerate(boxes)]
    faces = np.stack([align_crop(f, b) for f, b in zip(frames, boxes)])
    np.savez_compressed(out_path, frames=faces, face_found=found, frame_idx=frame_idx,
                        frame_time_sec=frame_time_sec, video_fps=np.float32(fps))
    return sample_id, path.name, int((~found).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, help="folder containing the RAVDESS .mp4 files")
    ap.add_argument("--out-dir", default="data/faces")
    ap.add_argument("--modality", default="01", choices=["01", "02"],
                    help="01 = audio-video, 02 = video-only (same frames); use the same as M4")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = find_videos(args.video_dir, args.modality)
    print(f"Found {len(videos)} speech videos (modality {args.modality})")

    todo = [(sid, p, out_dir / f"{sid}.npz") for sid, p in videos.items()
            if not (out_dir / f"{sid}.npz").exists()]  # resumable
    print(f"{len(videos) - len(todo)} already done, {len(todo)} to process")

    log_path = out_dir / "face_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as fh, \
            ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["sample_id", "file", "frames_without_face"])
        for sid, name, missing in tqdm(pool.map(process, todo, chunksize=4), total=len(todo)):
            if missing:
                writer.writerow([sid, name, missing])


if __name__ == "__main__":
    main()
