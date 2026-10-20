"""Tests for the train-only augmentation experiment (synthetic faces; real data only if present)."""
import csv
import hashlib
import json
import random

import joblib
import numpy as np
import pytest

from m3 import augment_faces as af
from m3 import dwt_features
from m3 import run_handcrafted_ablation as abl
from m3 import run_handcrafted_augmentation as aug
from m3.common import EMOTIONS, ROOT
from tests.test_handcrafted_ablation import file_hashes, synthetic_frames, write_faces

RUNTIME_COLS = ("feature_seconds", "svm_seconds", "runtime_seconds")
N_SYN = 40  # actors 1, 4, 5 (train), 9 (val), 2 (test) x 8 emotions
REAL_PROTECTED = ("outputs", "outputs_v1_global20", "outputs_ablation")


def tree_hashes(d):
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def real_protected_hashes():
    return {d: tree_hashes(ROOT / d) for d in REAL_PROTECTED if (ROOT / d).exists()}


def make_inputs(root, faces):
    """Baseline D.npz (ablation format) and sequence file for a synthetic faces dir."""
    files = dwt_features.face_files(faces)
    ids = [f.stem for f in files]
    labels, splits = abl.build_metadata(ids, abl.load_actor_splits())
    X, _ = abl.extract_features(files, {"D": (180, 4)})
    base = root / "D.npz"
    np.savez(base, sample_id=np.array(ids), split=splits, label=labels, F_pooled=X["D"],
             inner_crop=180, grid=4, feature_seconds=0.0)
    seq = root / "dwt_features.npz"
    dwt_features.main(["--faces-dir", str(faces), "--out", str(seq)])
    return base, seq


def run(env, out, *extra):
    return aug.main(["--faces-dir", str(env["faces"]), "--baseline-features", str(env["base"]),
                     "--sequence-features", str(env["seq"]), "--out-dir", str(out),
                     "--expected-samples", str(N_SYN), *extra])


def read_rows(out):
    with open(out / "augmentation_validation_results.csv") as fh:
        return list(csv.DictReader(fh))


def strip_runtime(rows):
    return [{k: v for k, v in r.items() if k not in RUNTIME_COLS} for r in rows]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("aug")
    faces = root / "faces"
    write_faces(faces)
    base, seq = make_inputs(root, faces)
    return {"root": root, "faces": faces, "base": base, "seq": seq}


@pytest.fixture(scope="module")
def run1(env):
    faces_before = file_hashes(env["faces"])
    protected_before = real_protected_hashes()
    out = env["root"] / "run1"
    rows, chosen, test = run(env, out)
    return {"out": out, "rows": rows, "chosen": chosen, "test": test,
            "faces_before": faces_before, "protected_before": protected_before}


def clip(seed=0):
    return synthetic_frames(3, seed)


SIDS = [f"01-{e:02d}-0{i}-01-0{r}-{a:02d}" for e in range(1, 9) for i in (1, 2)
        for r in (1, 2) for a in (1, 4)]


# ------------------------------------------------------------ augment_faces

# 1. one geometric parameter set for all 16 frames
def test_same_params_for_all_frames():
    frames = np.repeat(clip()[:1], 16, axis=0)  # 16 identical frames
    for sid in SIDS[:10]:
        out = af.augment_clip(frames, af.sample_params("AUG-2", sid, 0))
        assert all(np.array_equal(out[0], out[i]) for i in range(16))


# 2. same sample_id + config + seed -> same parameters (stable SHA-256, not hash())
def test_params_deterministic_and_stable_hash():
    sid = "01-03-01-01-01-05"
    assert af.sample_params("AUG-2", sid, 0) == af.sample_params("AUG-2", sid, 0)
    expected = int.from_bytes(hashlib.sha256(f"AUG-2|{sid}|0".encode()).digest()[:8], "little")
    assert af.local_seed("AUG-2", sid, 0) == expected


# 3. same inputs -> identical augmented images
def test_augmented_images_deterministic():
    p = af.sample_params("AUG-2", SIDS[3], 0)
    np.testing.assert_array_equal(af.augment_clip(clip(), p), af.augment_clip(clip(), p))


# 7. AUG-1 flips every frame horizontally
def test_aug1_flips_all_frames():
    frames = clip()
    out = af.augment_clip(frames, af.sample_params("AUG-1", SIDS[0], 0))
    np.testing.assert_array_equal(out, frames[:, :, ::-1])


def test_aug0_is_identity():
    frames = clip()
    np.testing.assert_array_equal(af.augment_clip(frames, af.sample_params("AUG-0", SIDS[0], 0)), frames)


# 8. AUG-2 parameters inside the ranges
def test_aug2_params_in_range():
    ps = [af.sample_params("AUG-2", s, seed) for s in SIDS for seed in (0, 1)]
    for p in ps:
        af.check_params(p)
        for k in ("rotation_deg", "translate_x", "translate_y", "scale", "brightness", "contrast"):
            lo, hi = af.AUG2_RANGES[k]
            assert lo <= p[k] <= hi
    flips = [p["flip"] for p in ps]
    assert any(flips) and not all(flips)


def test_out_of_range_params_rejected():
    with pytest.raises(ValueError):
        af.check_params({**af.IDENTITY, "rotation_deg": 10.0})
    with pytest.raises(ValueError):
        af.augment_clip(clip(), {**af.IDENTITY, "scale": 1.2})


# 9. a different global seed changes at least some parameters
def test_seed_changes_params():
    a = [af.sample_params("AUG-2", s, 0) for s in SIDS]
    b = [af.sample_params("AUG-2", s, 1) for s in SIDS]
    assert sum(x != y for x, y in zip(a, b)) > len(SIDS) // 2


# 10. input frames are not modified in place
@pytest.mark.parametrize("cfg", af.AUG_CONFIGS)
def test_input_not_modified(cfg):
    frames = clip()
    before = frames.copy()
    out = af.augment_clip(frames, af.sample_params(cfg, SIDS[5], 0))
    np.testing.assert_array_equal(frames, before)
    assert not np.shares_memory(out, frames)


# 11. output keeps uint8 [16, 224, 224, 3] in 0..255
@pytest.mark.parametrize("sid", SIDS[:6])
def test_output_format(sid):
    out = af.augment_clip(clip(), af.sample_params("AUG-2", sid, 0))
    assert out.shape == (16, 224, 224, 3) and out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_no_black_border():
    frames = np.full((16, 224, 224, 3), 200, np.uint8)
    p = {**af.IDENTITY, "rotation_deg": 5.0, "translate_x": 0.03, "translate_y": -0.03, "scale": 0.97}
    assert af.augment_clip(frames, p).min() == 200  # reflected border, no zeros


def test_bad_clip_rejected():
    with pytest.raises(ValueError):
        af.augment_clip(np.zeros((16, 200, 200, 3), np.uint8), af.IDENTITY)


# ------------------------------------------------- extraction determinism

def _items(env, n=6):
    files = dwt_features.face_files(env["faces"])
    return [(f.stem, f) for f in files[:n]]


# 4. workers=1 and workers>1 give the same features; 6. file order does not matter
def test_workers_and_order_do_not_change_features(env):
    items = _items(env)
    f1, _ = aug.extract_augmented(items, ("AUG-1", "AUG-2"), 0, workers=1)
    f2, _ = aug.extract_augmented(items, ("AUG-1", "AUG-2"), 0, workers=2)
    shuffled = items[:]
    random.Random(3).shuffle(shuffled)
    f3, _ = aug.extract_augmented(shuffled, ("AUG-1", "AUG-2"), 0, workers=1)
    for cfg in ("AUG-1", "AUG-2"):
        for sid, _ in items:
            for other in (f2, f3):
                np.testing.assert_array_equal(f1[cfg][sid][0], other[cfg][sid][0])
                assert f1[cfg][sid][1] == other[cfg][sid][1]
            assert f1[cfg][sid][0].shape == (320,)  # 18. 320-dim pooled features


# ---------------------------------------------------------- full pipeline

# 12. faces unchanged; 13. real protected outputs unchanged
def test_inputs_and_protected_outputs_unchanged(env, run1):
    assert file_hashes(env["faces"]) == run1["faces_before"]
    assert real_protected_hashes() == run1["protected_before"]


def test_output_files(run1):
    out = run1["out"]
    for f in ("augmentation_validation_results.csv", "augmentation_validation_results.json",
              "selected_augmentation.json", "test_metrics.json", "confusion_matrix.csv",
              "selected_svm.joblib", "augmentation_config.json", "features/AUG-1.npz",
              "features/AUG-2.npz", "m3_visual_handcrafted_augmented.npz"):
        assert (out / f).exists(), f
    rows = read_rows(out)
    assert [r["config_name"] for r in rows] == list(af.AUG_CONFIGS)
    assert set(aug.CSV_COLUMNS) <= set(rows[0])
    assert [int(r["total_train_samples"]) for r in rows] == [24, 48, 48]
    assert all(int(r["feature_dim"]) == 320 and int(r["seed"]) == 0 for r in rows)


# 14. / 15. no augmented validation or test samples
def test_only_train_augmented(run1):
    out = run1["out"]
    for cfg in ("AUG-1", "AUG-2"):
        z = np.load(out / "features" / f"{cfg}.npz")
        assert set(z["split"]) == {"train"}
        assert {s[-2:] for s in z["source_sample_id"]} == {"01", "04", "05"}  # train actors
        assert all(s.endswith(aug.AUG_SUFFIX) for s in z["augmented_sample_id"])
        assert z["F_pooled"].shape == (24, 320)
    rows = read_rows(out)
    assert all(int(r["validation_samples"]) == 8 for r in rows)
    assert not any("test" in k for k in rows[0])


# 16. AUG-0 reproduces the D baseline; 17. exported features equal D.npz
def test_aug0_reproduces_baseline(env, run1):
    base = np.load(env["base"])
    sp, y, X = base["split"], base["label"], base["F_pooled"]
    res, _ = abl.evaluate_config(X[sp == "train"], y[sp == "train"], X[sp == "val"], y[sp == "val"], 0)
    row = read_rows(run1["out"])[0]
    assert float(row["validation_macro_f1"]) == pytest.approx(res["validation_macro_f1"], abs=1e-12)
    assert float(row["validation_accuracy"]) == pytest.approx(res["validation_accuracy"], abs=1e-12)
    npz = np.load(run1["out"] / "m3_visual_handcrafted_augmented.npz")
    np.testing.assert_array_equal(npz["F_visual_handcrafted"], X)


# 19. / 25.-28. candidate export
def test_candidate_export(env, run1):
    npz = np.load(run1["out"] / "m3_visual_handcrafted_augmented.npz")
    base, seq = np.load(env["base"]), np.load(env["seq"])
    n = N_SYN
    assert not any(aug.AUG_SUFFIX in s for s in npz["sample_id"])
    np.testing.assert_array_equal(npz["sample_id"], base["sample_id"])
    assert len(set(npz["sample_id"])) == n
    for k in ("split", "label"):
        np.testing.assert_array_equal(npz[k], base[k])
    for k in ("F_sequence", "frame_idx", "frame_time_sec", "video_fps"):
        np.testing.assert_array_equal(npz[k], seq[k])
    assert npz["F_sequence"].shape == (n, 16, 320) and npz["padding_mask"].shape == (n, 16)
    assert npz["sequence_length"].tolist() == [16] * n
    P = npz["P_visual_handcrafted"]
    assert P.shape == (n, 8)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-4)
    assert list(npz["class_names"]) == EMOTIONS
    model = joblib.load(run1["out"] / "selected_svm.joblib")
    assert list(model.classes_) == list(range(8))
    np.testing.assert_allclose(model.predict_proba(npz["F_visual_handcrafted"]), P, atol=1e-6)
    assert str(npz["training_augmentation"]) == af.TRAINING_AUGMENTATION[run1["chosen"]]
    assert int(npz["augmentation_seed"]) == 0
    assert "aug2_ranges" in json.loads(str(npz["augmentation_config"]))


# 23. selected = highest validation Macro-F1; 21. test only for the selected configuration
def test_selection_and_single_test(run1):
    rows = read_rows(run1["out"])
    sel = json.loads((run1["out"] / "selected_augmentation.json").read_text())
    best = max(float(r["validation_macro_f1"]) for r in rows)
    first_best = next(r["config_name"] for r in rows if float(r["validation_macro_f1"]) == best)
    assert sel["selected_config"]["config_name"] == first_best == run1["chosen"]
    assert "tie" in sel["tie_breaking"].lower() and sel["svm_search_space"]
    tm = json.loads((run1["out"] / "test_metrics.json").read_text())
    assert tm["config_name"] == run1["chosen"] and tm["n_test"] == 8


def test_test_evaluated_exactly_once(env, tmp_path, monkeypatch):
    calls = []
    real = aug.evaluate_test_once
    monkeypatch.setattr(aug, "evaluate_test_once", lambda *a: calls.append(1) or real(*a))
    run(env, tmp_path / "once")
    assert len(calls) == 1


# 24. ties -> AUG-0, then AUG-1, then AUG-2
def test_tie_breaking():
    mk = lambda fs: [{"config_name": c, "validation_macro_f1": f} for c, f in zip(af.AUG_CONFIGS, fs)]
    assert abl.select_config(mk([0.4, 0.4, 0.4])) == "AUG-0"
    assert abl.select_config(mk([0.3, 0.5, 0.5])) == "AUG-1"
    assert abl.select_config(mk([0.3, 0.4, 0.5])) == "AUG-2"


# 20. test data does not reach configuration selection
def test_test_split_not_used_for_selection(env, run1, tmp_path):
    base, seq = dict(np.load(env["base"])), dict(np.load(env["seq"]))
    te = base["split"] == "test"
    rng = np.random.default_rng(7)
    seq["F_sequence"] = seq["F_sequence"].copy()
    seq["F_sequence"][te] = rng.normal(size=seq["F_sequence"][te].shape).astype(np.float32)
    base["F_pooled"] = seq["F_sequence"].mean(axis=1)
    np.savez(tmp_path / "D.npz", **base)
    np.savez(tmp_path / "seq.npz", **seq)
    env2 = {**env, "base": tmp_path / "D.npz", "seq": tmp_path / "seq.npz"}
    run(env2, tmp_path / "out")
    assert strip_runtime(read_rows(tmp_path / "out")) == strip_runtime(read_rows(run1["out"]))
    sel = lambda d: json.loads((d / "selected_augmentation.json").read_text())["selected_config"]["config_name"]
    assert sel(tmp_path / "out") == sel(run1["out"])


# 5. / 22. reproducible with the same seed, also with more workers
def test_reproducible_with_workers(env, run1, tmp_path):
    run(env, tmp_path / "w2", "--workers", "2")
    a, b = run1["out"], tmp_path / "w2"
    assert strip_runtime(read_rows(b)) == strip_runtime(read_rows(a))
    for f in ("test_metrics.json", "confusion_matrix.csv"):
        assert (b / f).read_text() == (a / f).read_text()
    for cfg in ("AUG-1", "AUG-2"):
        za, zb = np.load(a / "features" / f"{cfg}.npz"), np.load(b / "features" / f"{cfg}.npz")
        for k in ("source_sample_id", "augmented_sample_id", "F_pooled", "params_json"):
            np.testing.assert_array_equal(za[k], zb[k])
        assert list(za["source_sample_id"]) == sorted(za["source_sample_id"])
    pa = np.load(a / "m3_visual_handcrafted_augmented.npz")
    pb = np.load(b / "m3_visual_handcrafted_augmented.npz")
    for k in ("sample_id", "P_visual_handcrafted", "F_visual_handcrafted"):
        np.testing.assert_array_equal(pa[k], pb[k])


def test_reuse_cache_and_reject_bad_cache(env, run1, tmp_path):
    dst = tmp_path / "reuse"
    (dst / "features").mkdir(parents=True)
    for p in (run1["out"] / "features").glob("*.npz"):
        (dst / "features" / p.name).write_bytes(p.read_bytes())
    run(env, dst, "--reuse-features")
    assert strip_runtime(read_rows(dst)) == strip_runtime(read_rows(run1["out"]))
    cache = np.load(dst / "features" / "AUG-2.npz")
    ids, y = list(cache["source_sample_id"]), cache["label"]
    with pytest.raises(ValueError, match="seed"):  # cache was made with seed 0
        aug.load_aug_cache(dst / "features" / "AUG-2.npz", "AUG-2", 1, ids, y)
    with pytest.raises(ValueError, match="order"):
        aug.load_aug_cache(dst / "features" / "AUG-2.npz", "AUG-2", 0, ids[::-1], y[::-1])
    bad = dict(cache)
    bad["params_json"] = bad["params_json"].copy()
    bad["params_json"][0] = json.dumps({**af.IDENTITY, "rotation_deg": 4.0}, sort_keys=True)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        aug.load_aug_cache(tmp_path / "bad.npz", "AUG-2", 0, ids, y)


def test_bad_baseline_rejected(env, tmp_path):
    base = dict(np.load(env["base"]))
    ids = list(base["sample_id"])
    labels, splits = base["label"], base["split"]
    bad = {**base, "sample_id": np.array(ids[::-1])}
    np.savez(tmp_path / "order.npz", **bad)
    with pytest.raises(ValueError, match="order"):
        aug.load_baseline(tmp_path / "order.npz", ids, labels, splits)
    bad = {**base, "grid": 1}
    np.savez(tmp_path / "grid.npz", **bad)
    with pytest.raises(ValueError, match="config D"):
        aug.load_baseline(tmp_path / "grid.npz", ids, labels, splits)
    X = base["F_pooled"].copy()
    X[0, 0] = np.nan
    np.savez(tmp_path / "nan.npz", **{**base, "F_pooled": X})
    with pytest.raises(ValueError, match="NaN"):
        aug.load_baseline(tmp_path / "nan.npz", ids, labels, splits)


# 29. output-folder protection
@pytest.mark.parametrize("d", ["outputs", "outputs_v1_global20", "outputs_ablation", "data/faces"])
def test_protected_dirs_refused(env, d):
    with pytest.raises(SystemExit, match="Refusing"):
        run(env, ROOT / d)


def test_existing_results_and_smoke_into_formal_refused(env, run1):
    with pytest.raises(SystemExit, match="overwrite"):
        run(env, run1["out"])
    with pytest.raises(SystemExit, match="Smoke"):
        run(env, ROOT / aug.FORMAL_DIR, "--max-per-class", "1")


# 30. small real-data smoke run (skipped without the real data)
REAL = (ROOT / "data" / "faces").exists() and (ROOT / "outputs_ablation" / "features" / "D.npz").exists() \
    and (ROOT / "data" / "dwt_features.npz").exists()


@pytest.mark.skipif(not REAL, reason="real data not present")
def test_real_smoke(tmp_path):
    files = dwt_features.face_files(ROOT / "data" / "faces")
    ids = [f.stem for f in files]
    labels, splits = abl.build_metadata(ids, abl.load_actor_splits())
    keep = abl.select_subset(ids, labels, splits, 1)
    used = [files[i] for i in keep]
    before = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in used}
    protected = real_protected_hashes()
    aug.main(["--faces-dir", str(ROOT / "data" / "faces"),
              "--baseline-features", str(ROOT / "outputs_ablation" / "features" / "D.npz"),
              "--sequence-features", str(ROOT / "data" / "dwt_features.npz"),
              "--out-dir", str(tmp_path / "smoke"), "--max-per-class", "1"])
    assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in used} == before
    assert real_protected_hashes() == protected
    npz = np.load(tmp_path / "smoke" / "m3_visual_handcrafted_augmented.npz")
    assert npz["P_visual_handcrafted"].shape == (24, 8)
    assert json.loads((tmp_path / "smoke" / "test_metrics.json").read_text())["smoke_subset_max_per_class"] == 1


FORMAL = ROOT / aug.FORMAL_DIR / "augmentation_validation_results.csv"


@pytest.mark.skipif(not FORMAL.exists(), reason="formal augmentation run not present")
def test_formal_aug0_matches_ablation_baseline():
    row = read_rows(ROOT / aug.FORMAL_DIR)[0]
    d = json.loads((ROOT / "outputs_ablation" / "selected_config.json").read_text())["selected_config"]
    assert row["config_name"] == "AUG-0" and d["config_name"] == "D"
    assert float(row["validation_macro_f1"]) == pytest.approx(d["validation_macro_f1"], abs=1e-9)
    assert float(row["validation_accuracy"]) == pytest.approx(d["validation_accuracy"], abs=1e-9)
