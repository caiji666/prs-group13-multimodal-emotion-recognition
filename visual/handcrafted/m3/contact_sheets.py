"""Quality check: put the 16 face frames of each video into one image (4x4 grid).

Reads data/faces/<sample_id>.npz (read-only) and writes one JPEG per video:
    <out_dir>/<NN_emotion>/<sample_id>.jpg
Header: sample_id, emotion, intensity, actor, split. Each tile shows the frame
number and time; tiles where no face was detected get a red border.

Usage:
    uv run python -m m3.contact_sheets --faces-dir data/faces --out-dir data/contact_sheets
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from m3.common import EMOTIONS, actor_from_sample_id, label_from_sample_id, load_actor_splits

COLS = 4
TILE = 200
HEADER = 40
FONT = cv2.FONT_HERSHEY_SIMPLEX


def make_sheet(frames, face_found, frame_idx, frame_time_sec, title):
    """frames: uint8 [16, H, W, 3] RGB -> BGR uint8 image with a 4x4 grid."""
    rows = int(np.ceil(len(frames) / COLS))
    sheet = np.full((HEADER + rows * TILE, COLS * TILE, 3), 255, np.uint8)
    cv2.putText(sheet, title, (8, 27), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    for k, fr in enumerate(frames):
        tile = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), (TILE, TILE), interpolation=cv2.INTER_AREA)
        if not face_found[k]:
            cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (0, 0, 255), 4)
        label = f"#{k + 1} f{int(frame_idx[k])}"
        if frame_time_sec is not None:
            label += f" {float(frame_time_sec[k]):.2f}s"
        cv2.rectangle(tile, (0, TILE - 20), (TILE, TILE), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, TILE - 6), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(k, COLS)
        sheet[HEADER + r * TILE:HEADER + (r + 1) * TILE, c * TILE:(c + 1) * TILE] = tile
    return sheet


def title_for(sample_id, split):
    _, emo, inten, stmt, rep, actor = sample_id.split("-")
    return (f"{sample_id} | {EMOTIONS[int(emo) - 1]} | "
            f"{'strong' if inten == '02' else 'normal'} | stmt {stmt} rep {rep} | "
            f"actor {int(actor)} | {split}")


def process(args):
    path, out_dir, split = args
    sid = path.stem
    with np.load(path) as z:
        sheet = make_sheet(z["frames"], z["face_found"], z["frame_idx"],
                           z["frame_time_sec"] if "frame_time_sec" in z.files else None,
                           title_for(sid, split))
    label = label_from_sample_id(sid)
    folder = out_dir / f"{label + 1:02d}_{EMOTIONS[label]}"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{sid}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces-dir", default="data/faces")
    ap.add_argument("--out-dir", default="data/contact_sheets")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)

    splits = load_actor_splits()
    out_dir = Path(args.out_dir)
    files = sorted(p for p in Path(args.faces_dir).glob("*.npz") if not p.stem.endswith(".tmp"))
    jobs = [(f, out_dir, splits[actor_from_sample_id(f.stem)]) for f in files]
    with ProcessPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(process, jobs, chunksize=8), total=len(jobs)))
    print(f"Saved {len(jobs)} images -> {out_dir}")


if __name__ == "__main__":
    main()
