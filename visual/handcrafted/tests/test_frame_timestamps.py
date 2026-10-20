"""Unit tests for frame timing metadata (synthetic data, no real RAVDESS files needed)."""
import json

import cv2
import numpy as np
import pytest

from m3 import add_frame_timestamps, dwt_features, extract_faces, train_svm
from m3.common import EMOTIONS, check_frame_timing, frame_times

FPS = 25.0
N_VIDEO_FRAMES = 40


def write_video(path, fps=FPS, n=N_VIDEO_FRAMES, size=64):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(0)
    for _ in range(n):
        vw.write(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    vw.release()


def make_dataset(tmp_path, sample_ids):
    """Videos (modality 01) + face npz files in the OLD format (no timing fields)."""
    vdir, fdir = tmp_path / "videos", tmp_path / "faces"
    vdir.mkdir()
    fdir.mkdir()
    rng = np.random.default_rng(1)
    for sid in sample_ids:
        write_video(vdir / f"01-{sid}.mp4")
        np.savez_compressed(
            fdir / f"{sid}.npz",
            frames=rng.integers(0, 255, (16, 224, 224, 3), dtype=np.uint8),
            face_found=rng.random(16) > 0.2,
            frame_idx=np.linspace(0, N_VIDEO_FRAMES - 1, 16).round().astype(np.int32),
        )
    return vdir, fdir


SIDS = ["01-01-01-01-01-01", "01-03-02-01-02-02", "01-05-01-02-01-09", "01-08-02-02-02-17"]


# 1. frame_time_sec = frame_idx / video_fps
def test_frame_times_equals_idx_over_fps():
    idx = np.array([0, 7, 13, 20, 98])
    t = frame_times(idx, 29.97)
    assert t.dtype == np.float32
    np.testing.assert_allclose(t, idx / 29.97, rtol=1e-6)


# 2. invalid fps raises
@pytest.mark.parametrize("fps", [0, -30.0, float("nan"), float("inf")])
def test_invalid_fps_raises(fps):
    with pytest.raises(ValueError):
        frame_times(np.arange(16), fps)


@pytest.mark.parametrize("idx", [np.array([0, 5, 3]), np.array([-1, 2, 4])])
def test_invalid_frame_idx_raises(idx):
    with pytest.raises(ValueError):
        frame_times(idx, 30.0)


def test_check_frame_timing_detects_mismatch():
    idx = np.arange(16)
    check_frame_timing(idx, frame_times(idx, 30.0), 30.0)
    with pytest.raises(ValueError):
        check_frame_timing(idx, frame_times(idx, 25.0), 30.0)


def test_extract_faces_saves_timing(tmp_path):
    video = tmp_path / "01-01-01-01-01-01-01.mp4"
    write_video(video)
    out = tmp_path / "01-01-01-01-01-01.npz"
    extract_faces._init_worker()
    extract_faces.process(("01-01-01-01-01-01", video, out))
    d = np.load(out)
    assert d["frames"].shape == (16, 224, 224, 3)
    assert d["frame_time_sec"].dtype == np.float32 and d["frame_time_sec"].shape == (16,)
    assert d["video_fps"].dtype == np.float32 and d["video_fps"].shape == ()
    np.testing.assert_allclose(d["frame_time_sec"], d["frame_idx"] / float(d["video_fps"]), rtol=1e-6)


# 3. metadata backfill does not change frames / face_found / frame_idx
def test_backfill_preserves_original_arrays(tmp_path):
    vdir, fdir = make_dataset(tmp_path, SIDS)
    before = {s: dict(np.load(fdir / f"{s}.npz")) for s in SIDS}

    counts = add_frame_timestamps.run(vdir, fdir, "01", expected=len(SIDS))
    assert counts == {"updated": len(SIDS), "skipped": 0, "failed": 0}

    for s in SIDS:
        d = np.load(fdir / f"{s}.npz")
        for k in ("frames", "face_found", "frame_idx"):
            assert d[k].dtype == before[s][k].dtype
            np.testing.assert_array_equal(d[k], before[s][k])
        assert float(d["video_fps"]) == pytest.approx(FPS, rel=1e-3)
        np.testing.assert_allclose(d["frame_time_sec"], d["frame_idx"] / float(d["video_fps"]), rtol=1e-6)

    # idempotent: second run verifies and skips
    assert add_frame_timestamps.run(vdir, fdir, "01", expected=len(SIDS)) == \
        {"updated": 0, "skipped": len(SIDS), "failed": 0}


def test_backfill_errors(tmp_path):
    vdir, fdir = make_dataset(tmp_path, SIDS)
    with pytest.raises(ValueError, match="expected"):
        add_frame_timestamps.run(vdir, fdir, "01", expected=1440)
    (vdir / f"01-{SIDS[0]}.mp4").unlink()
    with pytest.raises(SystemExit):
        add_frame_timestamps.run(vdir, fdir, "01", expected=len(SIDS))


# 4. dwt_features.npz aggregates the timing fields
def test_dwt_features_aggregates_timing(tmp_path):
    vdir, fdir = make_dataset(tmp_path, SIDS)
    add_frame_timestamps.run(vdir, fdir, "01", expected=len(SIDS))
    out = tmp_path / "dwt.npz"
    dwt_features.main(["--faces-dir", str(fdir), "--out", str(out)])
    d = np.load(out)
    n = len(SIDS)
    assert d["frame_idx"].shape == (n, 16)
    assert d["frame_time_sec"].shape == (n, 16) and d["frame_time_sec"].dtype == np.float32
    assert d["video_fps"].shape == (n,) and d["video_fps"].dtype == np.float32
    for i, s in enumerate(d["sample_id"]):
        f = np.load(fdir / f"{s}.npz")
        np.testing.assert_array_equal(d["frame_idx"][i], f["frame_idx"])
        np.testing.assert_array_equal(d["frame_time_sec"][i], f["frame_time_sec"])
    assert d["F_sequence"].shape == (n, 16, 320)


def test_dwt_features_requires_timing(tmp_path):
    _, fdir = make_dataset(tmp_path, SIDS)  # old-format files, no timing
    with pytest.raises(KeyError, match="add_frame_timestamps"):
        dwt_features.main(["--faces-dir", str(fdir), "--out", str(tmp_path / "x.npz")])


# 5. final export passes the timing fields through unchanged
def test_train_svm_passes_timing_through(tmp_path):
    rng = np.random.default_rng(0)
    ids, labels, splits = [], [], []
    split_of = {a: s for s, actors in {"train": [1, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 18, 19, 20, 23, 24],
                                       "val": [9, 14, 21, 22], "test": [2, 3, 8, 17]}.items() for a in actors}
    for a in range(1, 25):
        for e in range(8):
            ids.append(f"01-{e + 1:02d}-01-01-01-{a:02d}")
            labels.append(e)
            splits.append(split_of[a])
    n = len(ids)
    y = np.array(labels)
    seq = (rng.normal(size=(n, 16, 6)) + y[:, None, None]).astype(np.float32)
    idx = np.tile(np.linspace(0, 99, 16).round().astype(np.int32), (n, 1))
    fps = np.full(n, 29.97, np.float32)
    feats = tmp_path / "feats.npz"
    np.savez(feats, sample_id=np.array(ids), split=np.array(splits), label=y,
             F_sequence=seq, F_pooled=seq.mean(1), feature_names=np.array([f"f{i}" for i in range(6)]),
             frame_idx=idx, frame_time_sec=(idx / fps[:, None]).astype(np.float32), video_fps=fps)

    train_svm.main(["--features", str(feats), "--out-dir", str(tmp_path / "out")])
    src, out = np.load(feats), np.load(tmp_path / "out" / "m3_visual_handcrafted.npz")
    for k in ("frame_idx", "frame_time_sec", "video_fps"):
        assert out[k].dtype == src[k].dtype
        np.testing.assert_array_equal(out[k], src[k])
    np.testing.assert_array_equal(out["sample_id"], src["sample_id"])
    assert list(out["class_names"]) == EMOTIONS
    assert json.loads((tmp_path / "out" / "metrics.json").read_text())["counts"]["train"] == 128
