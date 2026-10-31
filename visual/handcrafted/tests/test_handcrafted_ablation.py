"""Tests for the inner-crop x grid ablation (synthetic faces; real data only if present)."""
import csv
import hashlib
import json

import cv2
import numpy as np
import pytest
import pywt

from m3 import run_handcrafted_ablation as abl
from m3.common import ROOT, frame_times, load_actor_splits
from m3.dwt_features import (HIST_BINS, clip_features, edge, entropy, frame_features,
                             subband_size, validate_config)

EXPECTED_DIM = {"A": 20, "B": 20, "C": 320, "D": 320}
# 3 train actors, 1 val actor, 1 test actor (from actor_splits.csv) x 8 emotions = 40 samples
ACTORS = [1, 4, 5, 9, 2]
RUNTIME_COLS = ("feature_seconds", "svm_seconds", "runtime_seconds")


def synthetic_frames(label, seed):
    """16 noisy 224x224 faces with a bright patch whose position depends on the label."""
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 120, (16, 224, 224, 3), dtype=np.uint8)
    r, c = divmod(label, 4)
    frames[:, 40 + r * 70:90 + r * 70, 20 + c * 48:60 + c * 48] += 120
    return frames


def write_faces(fdir, actors=ACTORS, seed_offset=0, only_actor=None):
    fdir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, 99, 16).round().astype(np.int32)
    for a in actors:
        if only_actor is not None and a != only_actor:
            continue
        for e in range(8):
            sid = f"01-{e + 1:02d}-01-01-01-{a:02d}"
            np.savez_compressed(fdir / f"{sid}.npz",
                                frames=synthetic_frames(e, 1000 * a + e + seed_offset),
                                face_found=np.ones(16, bool), frame_idx=idx,
                                frame_time_sec=frame_times(idx, 30.0), video_fps=np.float32(30.0))


def file_hashes(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.glob("*.npz"))}


def read_rows(out_dir):
    with open(out_dir / "ablation_validation_results.csv") as fh:
        return list(csv.DictReader(fh))


def strip_runtime(rows):
    return [{k: v for k, v in r.items() if k not in RUNTIME_COLS} for r in rows]


@pytest.fixture(scope="module")
def faces(tmp_path_factory):
    fdir = tmp_path_factory.mktemp("abl") / "faces"
    write_faces(fdir)
    return fdir


@pytest.fixture(scope="module")
def run1(faces, tmp_path_factory):
    before = file_hashes(faces)
    out = tmp_path_factory.mktemp("abl_out") / "run1"
    abl.main(["--faces-dir", str(faces), "--out-dir", str(out)])
    return out, before


# 1. feature dimensions of the four configurations
@pytest.mark.parametrize("name", list(abl.CONFIGS))
def test_config_feature_dims(name):
    crop, grid = abl.CONFIGS[name]
    seq = clip_features(synthetic_frames(3, 0), grid, crop)
    assert seq.shape == (16, EXPECTED_DIM[name]) and seq.dtype == np.float32


# 2. grid=1, inner_crop=224 reproduces v1 (whole-subband statistics, 20 per frame)
def test_grid1_crop224_matches_v1_definition():
    rgb = synthetic_frames(5, 1)[0]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    LL, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    expected = []
    for b in (LL, LH, HL, HH):
        b = b.astype(np.float32)
        expected += [b.mean(), b.var(), (b ** 2).mean(), entropy(b), edge(b)]
    np.testing.assert_array_equal(frame_features(rgb, grid=1, inner_crop=224),
                                  np.array(expected, np.float32))
    assert HIST_BINS == 64


V1 = ROOT / "data" / "dwt_features_v1_global20.npz"
V2 = ROOT / "data" / "dwt_features.npz"


@pytest.mark.skipif(not (V1.exists() and V2.exists() and (ROOT / "data" / "faces").exists()),
                    reason="real features not present")
@pytest.mark.parametrize("i", [0, 777, 1439])
def test_real_samples_reproduce_v1_and_v2(i):
    v1, v2 = np.load(V1), np.load(V2)
    sid = v1["sample_id"][i]
    frames = np.load(ROOT / "data" / "faces" / f"{sid}.npz")["frames"]
    np.testing.assert_array_equal(clip_features(frames, 1, 224), v1["F_sequence"][i])
    np.testing.assert_array_equal(clip_features(frames, 4, 180), v2["F_sequence"][i])


# 3. grid=4, inner_crop=180 -> 4 subbands of 90x90, 320 features per frame
def test_grid4_crop180_is_320():
    assert subband_size(180) == 90 and subband_size(224) == 112
    assert frame_features(synthetic_frames(0, 2)[0], grid=4, inner_crop=180).shape == (320,)


# parameter validation
@pytest.mark.parametrize("crop,grid", [(0, 1), (-5, 1), (225, 1), (180, 0), (180, -2),
                                       (180, 2.0), (180.0, 4), (True, 1), (8, 4), (224, 40)])
def test_invalid_config_rejected(crop, grid):
    with pytest.raises(ValueError):
        validate_config(crop, grid)


def test_wrong_frame_size_rejected():
    with pytest.raises(ValueError, match="224x224x3"):
        frame_features(np.zeros((200, 200, 3), np.uint8))


def test_duplicate_and_misaligned_rejected():
    splits = load_actor_splits()
    with pytest.raises(ValueError, match="duplicate"):
        abl.build_metadata(["01-01-01-01-01-01"] * 2, splits)
    with pytest.raises(ValueError, match="sample_id"):
        abl.build_metadata(["03-01-01-01-01-01-01"], splits)
    ids = ["01-01-01-01-01-01", "01-02-01-01-01-09", "01-03-01-01-01-02"]
    labels, sp = abl.build_metadata(ids, splits)
    with pytest.raises(ValueError, match="row counts"):
        abl.check_aligned(ids, labels, sp, np.zeros((2, 20)))
    abl.check_aligned(ids, labels, sp, np.zeros((3, 20)))


def test_protected_output_dirs_refused(faces):
    for d in ("outputs", "outputs_v1_global20", "data/faces"):
        with pytest.raises(SystemExit, match="Refusing"):
            abl.main(["--faces-dir", str(faces), "--out-dir", str(ROOT / d)])


# 4. full run: outputs, dims, original frames untouched
def test_run_outputs_and_frames_untouched(faces, run1):
    out, before = run1
    assert file_hashes(faces) == before  # shared face files are read only
    for f in ("ablation_validation_results.csv", "ablation_validation_results.json",
              "selected_config.json", "test_metrics.json", "confusion_matrix.csv",
              "selected_svm.joblib"):
        assert (out / f).exists(), f
    rows = read_rows(out)
    assert [r["config_name"] for r in rows] == ["A", "B", "C", "D"]
    for r in rows:
        assert set(abl.CSV_COLUMNS) <= set(r)
        assert int(r["feature_dim_per_frame"]) == int(r["pooled_feature_dim"]) == EXPECTED_DIM[r["config_name"]]
        assert not any("test" in k for k in r)
    sel = json.loads((out / "selected_config.json").read_text())["selected_config"]
    best_val = max(float(r["validation_macro_f1"]) for r in rows)
    assert sel["validation_macro_f1"] == pytest.approx(best_val)
    assert sel["config_name"] == next(r["config_name"] for r in rows
                                      if float(r["validation_macro_f1"]) == pytest.approx(best_val))
    test = json.loads((out / "test_metrics.json").read_text())
    assert test["config_name"] == sel["config_name"] and test["n_test"] == 8
    with open(out / "confusion_matrix.csv") as fh:
        cm = list(csv.reader(fh))
    assert len(cm) == 9 and sum(int(v) for row in cm[1:] for v in row[1:]) == 8


def test_existing_results_not_overwritten(faces, run1):
    out, _ = run1
    with pytest.raises(SystemExit, match="overwrite"):
        abl.main(["--faces-dir", str(faces), "--out-dir", str(out)])


# 5. the test split does not take part in configuration selection
def test_test_split_not_used_for_selection(faces, run1, tmp_path):
    out, _ = run1
    other = tmp_path / "faces_other_test"
    other.mkdir()
    for p in faces.glob("*.npz"):
        (other / p.name).write_bytes(p.read_bytes())
    write_faces(other, seed_offset=99, only_actor=2)  # actor 2 = test: new random frames
    assert file_hashes(other) != file_hashes(faces)
    abl.main(["--faces-dir", str(other), "--out-dir", str(tmp_path / "run_other")])
    assert strip_runtime(read_rows(tmp_path / "run_other")) == strip_runtime(read_rows(out))
    sel = lambda d: json.loads((d / "selected_config.json").read_text())["selected_config"]["config_name"]
    assert sel(tmp_path / "run_other") == sel(out)


def test_select_config_uses_validation_only_and_breaks_ties_in_order():
    rows = [{"config_name": n, "validation_macro_f1": f}
            for n, f in zip("ABCD", [0.3, 0.5, 0.5, 0.4])]
    assert abl.select_config(rows) == "B"


# 6. same seed -> same results and same sample order
def test_reproducible(faces, run1, tmp_path):
    out, _ = run1
    abl.main(["--faces-dir", str(faces), "--out-dir", str(tmp_path / "run2")])
    assert strip_runtime(read_rows(tmp_path / "run2")) == strip_runtime(read_rows(out))
    for f in ("test_metrics.json", "confusion_matrix.csv"):
        assert (tmp_path / "run2" / f).read_text() == (out / f).read_text()
    for n in abl.CONFIGS:
        a, b = np.load(out / "features" / f"{n}.npz"), np.load(tmp_path / "run2" / "features" / f"{n}.npz")
        np.testing.assert_array_equal(a["sample_id"], b["sample_id"])
        assert list(a["sample_id"]) == sorted(a["sample_id"])
        np.testing.assert_array_equal(a["F_pooled"], b["F_pooled"])


def test_reuse_features_gives_same_results(faces, run1, tmp_path):
    out, _ = run1
    dst = tmp_path / "reuse"
    (dst / "features").mkdir(parents=True)
    for p in (out / "features").glob("*.npz"):
        (dst / "features" / p.name).write_bytes(p.read_bytes())
    abl.main(["--faces-dir", str(faces), "--out-dir", str(dst), "--reuse-features"])
    assert strip_runtime(read_rows(dst)) == strip_runtime(read_rows(out))


def test_smoke_subset_is_deterministic(faces, tmp_path):
    abl.main(["--faces-dir", str(faces), "--out-dir", str(tmp_path / "s"), "--max-per-class", "1"])
    z = np.load(tmp_path / "s" / "features" / "A.npz")
    # first actor of each split x emotion in sorted sample_id order: actors 01 (train), 09 (val), 02 (test)
    assert sorted({s[-2:] for s in z["sample_id"]}) == ["01", "02", "09"]
    assert len(z["sample_id"]) == 24
