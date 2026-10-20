"""Core tests for the Level-2 A / B / A+B comparison (synthetic arrays in tmp_path; real M3 read-only).

Grouped by what makes the experiment trustworthy: inputs and alignment, model selection and
leakage, actors / AUG-1 / OOF, and the final evaluation outputs.
"""
import csv
import hashlib
import json

import joblib
import numpy as np
import pytest
import sklearn.svm

from m3 import run_visual_representation_comparison as rvc
from m3.common import EMOTIONS, ROOT
from m3.train_svm import make_model, scores

COUNTS = {"train": 64, "val": 16, "test": 16}  # 8 / 2 / 2 per class
COUNTS_ARG = "train=64,val=16,test=16"
ACTORS = {"train": [1, 4, 5, 6], "val": [9, 14], "test": [2, 3]}  # actor-independent
ACTORS_ARG = "train=4,val=2,test=2"
GENDERS_ARG = "train=2:2,val=1:1,test=1:1"  # odd = male: train 1, 5 / 4, 6; val 9 / 14; test 3 / 2
H, D = 20, 24
TAGS = ("handcrafted", "deep", "combined")
PROB_COLS = ["prob_neutral", "prob_calm", "prob_happy", "prob_sad", "prob_angry", "prob_fearful",
             "prob_disgust", "prob_surprised"]


# ------------------------------------------------------------ helpers

def synthetic_ids():
    """4 train actors x 8 emotions x 2 repetitions; 2 val and 2 test actors x 8 emotions."""
    ids, labels, splits = [], [], []
    for s, actors in ACTORS.items():
        for a in actors:
            for e in range(8):
                for rep in range(2 if s == "train" else 1):
                    ids.append(f"01-{e + 1:02d}-01-01-{rep + 1:02d}-{a:02d}")
                    labels.append(e)
                    splits.append(s)
    order = np.argsort(ids)
    return np.array(ids)[order], np.array(labels)[order], np.array(splits)[order]


def save_aug1(path, ids, y, F_aug, **override):
    n = len(ids)
    d = {"source_sample_id": ids, "augmented_sample_id": np.array([s + "__aug01" for s in ids]),
         "label": y, "split": np.full(n, "train"), "F_pooled": F_aug, "config_name": "AUG-1",
         "seed": 0, "feature_dim": F_aug.shape[1], "inner_crop": 180, "grid": 4,
         "param_flip": np.ones(n, bool), "param_rotation_deg": np.zeros(n),
         "param_translate_x": np.zeros(n), "param_translate_y": np.zeros(n),
         "param_scale": np.ones(n), "param_brightness": np.ones(n), "param_contrast": np.ones(n)}
    d.update(override)
    np.savez(path, **d)
    return path


def make_m3(root, seed=0, labels_override=None):
    """Synthetic M3 export + outputs_augmentation-like folder (AUG-1 selected)."""
    ids, y, sp = synthetic_ids()
    if labels_override is not None:
        y = labels_override
    rng = np.random.default_rng(seed)
    n = len(ids)
    seq = (rng.normal(size=(n, 16, H)) + rng.normal(size=(n, 1, H))
           + y[:, None, None] * 0.15).astype(np.float32)
    F = seq.mean(axis=1)
    tr, va, te = sp == "train", sp == "val", sp == "test"
    F_aug = (F[tr] + rng.normal(scale=0.1, size=F[tr].shape)).astype(np.float32)  # "flipped" copies
    model = make_model(10, "scale", None).fit(np.vstack([F[tr], F_aug]), np.concatenate([y[tr], y[tr]]))
    aug = root / "aug"
    (aug / "features").mkdir(parents=True)
    save_aug1(aug / "features" / "AUG-1.npz", ids[tr], y[tr], F_aug)
    hand_path = aug / "m3_visual_handcrafted_augmented.npz"
    idx = np.tile(np.arange(16, dtype=np.int32), (n, 1))
    np.savez(hand_path, sample_id=ids, split=sp, label=y, F_sequence=seq,
             sequence_length=np.full(n, 16), padding_mask=np.zeros((n, 16), bool),
             F_visual_handcrafted=F, P_visual_handcrafted=model.predict_proba(F).astype(np.float32),
             class_names=np.array(EMOTIONS), feature_names=np.array([f"f{i}" for i in range(H)]),
             frame_idx=idx, frame_time_sec=(idx / 30).astype(np.float32),
             video_fps=np.full(n, 30, np.float32), training_augmentation="horizontal_flip",
             augmentation_seed=0, augmentation_config="{}")
    from sklearn.metrics import accuracy_score, f1_score
    pred_va = model.predict(F[va])
    with open(aug / "augmentation_validation_results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config_name", "original_train_samples", "augmented_train_samples",
                    "total_train_samples", "best_C", "best_gamma", "best_class_weight",
                    "validation_macro_f1", "validation_accuracy"])
        w.writerow(["AUG-1", 64, 64, 128, 10, "scale", "",
                    f1_score(y[va], pred_va, average="macro"), accuracy_score(y[va], pred_va)])
    (aug / "selected_augmentation.json").write_text(json.dumps(
        {"selected_config": {"config_name": "AUG-1"}, "protocol": {"seed": 0}}))
    test = scores(y[te], model.predict(F[te]))
    (aug / "test_metrics.json").write_text(json.dumps(
        {"config_name": "AUG-1", "n_test": int(te.sum()), "test": test}, default=str))
    rvc.write_cm(aug / "confusion_matrix.csv", test["confusion_matrix"])
    joblib.dump(model, aug / "selected_svm.joblib")
    return {"hand": hand_path, "aug": aug, "ids": ids, "y": y, "split": sp, "F": F, "F_aug": F_aug}


def save_deep(path, m3, seed=1, drop=(), shuffle=False, **override):
    rng = np.random.default_rng(seed)
    n = len(m3["ids"])
    F = (rng.normal(size=(n, D)) + m3["y"][:, None] * 0.8).astype(np.float32)
    d = {"sample_id": m3["ids"], "split": m3["split"], "label": m3["y"],
         "class_names": np.array(EMOTIONS), "F_visual_deep": F,
         "F_visual_sequence": np.repeat(F[:, None, :8], 16, axis=1),
         "P_visual_deep": np.full((n, 8), 1 / 8, np.float32), "source_model": "TEST_MODEL"}
    d.update(override)
    for k in drop:
        d.pop(k)
    if shuffle:
        perm = rng.permutation(n)
        d = {k: (v[perm] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == n else v)
             for k, v in d.items()}
    np.savez(path, **d)
    return path


def argv(m3, deep, out, *extra):
    return ["--deep-features", str(deep), "--handcrafted", str(m3["hand"]),
            "--augmentation-dir", str(m3["aug"]), "--out-dir", str(out),
            "--expected-counts", COUNTS_ARG, "--expected-actor-counts", ACTORS_ARG,
            "--expected-gender-counts", GENDERS_ARG, *extra]


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def no_time(rows):
    return [{k: v for k, v in r.items() if k not in ("fit_seconds", "source")} for r in rows]


def hand(m3):
    return rvc.load_handcrafted(m3["hand"], COUNTS)


def oof_inputs(m3):
    h = hand(m3)
    aug = rvc.load_aug1_features(m3["aug"] / "features" / "AUG-1.npz", h, 0)
    fold, folds = rvc.make_actor_grouped_folds(h["sample_id"], h["split"])
    return h, aug, fold, folds


@pytest.fixture(scope="module")
def m3(tmp_path_factory):
    return make_m3(tmp_path_factory.mktemp("lvl2"))


@pytest.fixture(scope="module")
def run1(m3, tmp_path_factory):
    root = tmp_path_factory.mktemp("lvl2run")
    deep = save_deep(root / "deep.npz", m3)
    out = root / "out"
    status = rvc.main(argv(m3, deep, out))
    return {"out": out, "deep": deep, "status": status}


REAL_HAND = ROOT / rvc.DEFAULT_HANDCRAFTED


# ================================================ A. inputs and alignment

@pytest.mark.skipif(not REAL_HAND.exists(), reason="real M3 export not present")
def test_real_m3_input():
    """Real M3 export: 1440 unique clips, 960/240/240, class order, finite features and probabilities."""
    h = rvc.load_handcrafted(REAL_HAND)  # raises on duplicates, wrong counts, NaN, bad probabilities
    assert len(set(h["sample_id"])) == 1440
    assert {s: int((h["split"] == s).sum()) for s in ("train", "val", "test")} == rvc.EXPECTED_COUNTS
    assert h["F"].ndim == 2 and h["F_sequence"].shape[:2] == (1440, 16)
    assert list(h["class_names"]) == EMOTIONS
    assert np.isfinite(h["F"]).all() and np.isfinite(h["P"]).all()


def test_waiting_for_m4(m3, tmp_path, monkeypatch, capsys):
    """Missing M4 file: WAITING_FOR_M4, no training, no output or staging folder."""
    monkeypatch.setattr(rvc, "search_mlp", lambda *a, **k: pytest.fail("no training while waiting"))
    out = tmp_path / "out"
    st = rvc.main(argv(m3, tmp_path / "missing.npz", out, "--validate-only"))
    assert st["status"] == "WAITING_FOR_M4" and "WAITING_FOR_M4" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="WAITING_FOR_M4"):
        rvc.main(argv(m3, tmp_path / "missing.npz", out))
    assert not out.exists() and not (tmp_path / ".out.partial").exists()


def test_m4_rows_aligned_by_sample_id(m3, tmp_path):
    """Shuffled M4 rows are reordered by sample_id; the result equals the unshuffled input."""
    h = hand(m3)
    a = rvc.load_deep(save_deep(tmp_path / "a.npz", m3), h)
    b = rvc.load_deep(save_deep(tmp_path / "b.npz", m3, shuffle=True), h)
    assert a["reordered"] is False and b["reordered"] is True
    for k in ("F", "P", "sequence"):
        np.testing.assert_array_equal(a[k], b[k])
    # the combined feature follows the M3 sample order, not the M4 row order
    np.testing.assert_array_equal(rvc.build_combined(h["F"], b["F"])[:, H:], a["F"])


def _ids_with(m3, i, value, dtype=None):
    ids = m3["ids"].astype(dtype) if dtype else m3["ids"].copy()
    ids[i] = value if not callable(value) else value(ids)
    return ids


INVALID_M4 = {
    "missing_field": (lambda m3: dict(drop=("F_visual_deep",)), "missing required"),
    "duplicate_id": (lambda m3: dict(sample_id=_ids_with(m3, 1, lambda ids: ids[0])), "duplicate"),
    "id_set_differs": (lambda m3: dict(sample_id=_ids_with(m3, 3, "01-01-02-02-02-99")), "sets differ"),
    "augmented_id": (lambda m3: dict(sample_id=_ids_with(m3, 0, lambda ids: ids[0] + "__aug01", "U32")),
                     "augmented"),
    "label_mismatch": (lambda m3: dict(label=np.where(np.arange(len(m3["y"])) == 5, (m3["y"] + 1) % 8, m3["y"])),
                       "label mismatch"),
    "split_mismatch": (lambda m3: dict(split=np.where(np.arange(len(m3["ids"])) == int(np.flatnonzero(
        m3["split"] == "train")[0]), "val", m3["split"])), "split mismatch"),  # one sample_id changed
    "class_order": (lambda m3: dict(class_names=np.array(EMOTIONS[::-1])), "class_names"),
    "nan_feature": (lambda m3: dict(F_visual_deep=np.full((len(m3["ids"]), D), np.nan, np.float32)), "NaN"),
    "feature_rows": (lambda m3: dict(F_visual_deep=np.zeros((len(m3["ids"]) - 1, D), np.float32)), "2-D"),
    "prob_sum": (lambda m3: dict(P_visual_deep=np.full((len(m3["ids"]), 8), 0.2)), "sum to 1"),
    "sequence_frames": (lambda m3: dict(F_visual_sequence=np.zeros((len(m3["ids"]), 12, 8))), "16"),
    "optional_missing_ok": (lambda m3: dict(drop=("F_visual_sequence", "P_visual_deep")), None),
}


@pytest.mark.parametrize("case", list(INVALID_M4))
def test_invalid_m4_input(m3, tmp_path, case):
    """Invalid M4 files are rejected with a readable error; optional fields may be absent."""
    kwargs, match = INVALID_M4[case]
    path = save_deep(tmp_path / f"{case}.npz", m3, **kwargs(m3))
    if match is None:
        d = rvc.load_deep(path, hand(m3))
        assert d["sequence"] is None and d["P"] is None
    else:
        with pytest.raises(rvc.DataError, match=match):
            rvc.load_deep(path, hand(m3))


def test_source_model_rules(m3, tmp_path, capsys):
    """source_model: from the NPZ, or from --deep-source-model; conflict and absence stop the run."""
    with_model = save_deep(tmp_path / "a.npz", m3)
    without = save_deep(tmp_path / "b.npz", m3, drop=("source_model",))
    assert rvc.main(argv(m3, with_model, tmp_path / "o", "--validate-only"))["source_model"] == "TEST_MODEL"
    assert rvc.main(argv(m3, without, tmp_path / "o", "--validate-only",
                         "--deep-source-model", "X"))["source_model"] == "X"
    with pytest.raises(SystemExit, match="conflict"):
        rvc.main(argv(m3, with_model, tmp_path / "o", "--deep-source-model", "OTHER"))
    assert rvc.main(argv(m3, without, tmp_path / "o", "--validate-only"))["source_model"] is None
    with pytest.raises(SystemExit, match="source_model missing"):
        rvc.main(argv(m3, without, tmp_path / "o"))
    assert not (tmp_path / "o").exists()


# ============================== B. model training, selection and leakage

def test_same_mlp_protocol_for_b_and_ab(run1):
    """B and A+B use the same 9 MLP settings and seed; each picks its best on validation Macro-F1."""
    rows = read_csv(run1["out"] / "validation_results.csv")
    grid = {r: [(x["hidden_layer_sizes"], x["alpha"], x["random_seed"], x["total_train_samples"])
                for x in rows if x["representation"] == r] for r in ("B", "A+B")}
    assert len(grid["B"]) == 9 and grid["B"] == grid["A+B"]
    for r in ("B", "A+B"):
        cand = [x for x in rows if x["representation"] == r]
        best = next(x for x in cand if x["selected_config"] == "True")
        assert float(best["validation_macro_f1"]) == max(float(x["validation_macro_f1"]) for x in cand)
    assert rvc.MLP_FIXED["early_stopping"] and len(rvc.MLP_GRID) == 9


def test_scaler_fit_on_train_only(m3, run1):
    """Locked B / A+B artifacts: scaler fitted on train rows only, plus the self-describing metadata."""
    deep = rvc.load_deep(run1["deep"], hand(m3))
    tr = m3["split"] == "train"
    sc = json.loads((run1["out"] / "selected_configs.json").read_text())["configurations"]
    order = ["F_visual_handcrafted", "F_visual_deep"]
    expected = {"best_deep_mlp.joblib": ("B", "deep", D, ["F_visual_deep"], deep["F"]),
                "best_combined_mlp.joblib": ("A+B", "combined", H + D, order,
                                             rvc.build_combined(m3["F"], deep["F"]))}
    for name, (rep, rep_name, dim, fields, X) in expected.items():
        art = joblib.load(run1["out"] / name)
        assert (art["representation"], art["representation_name"], art["feature_dim"], art["input_fields"]) == \
            (rep, rep_name, dim, fields)
        assert art["class_names"] == EMOTIONS and art["random_seed"] == 0
        assert art["source_model"] == "TEST_MODEL" and "source_checkpoint" in art
        best = sc[rep]["best_mlp"]
        assert list(art["selected_hidden_layers"]) == list(best["hidden_layer_sizes"])
        assert art["selected_alpha"] == best["alpha"]
        assert not any(isinstance(v, np.ndarray) for k, v in art.items() if k != "model")  # no feature arrays
        scaler = art["model"][0]
        assert scaler.n_samples_seen_ == tr.sum()
        np.testing.assert_allclose(scaler.mean_, X[tr].mean(axis=0), rtol=1e-6, atol=1e-6)
        assert art["model"].predict_proba(X[:3]).shape == (3, 8)
        if rep == "A+B":
            assert art["concatenation_order"] == order and art["concatenation_axis"] == 1
            assert (art["handcrafted_feature_dim"], art["deep_feature_dim"]) == (H, D)
            assert art["handcrafted_feature_dim"] + art["deep_feature_dim"] == art["feature_dim"]


def test_test_rows_do_not_affect_selection(run1, tmp_path):
    """Changing test features and test labels changes neither the MLP choices, the selection nor validation scores."""
    ids, y, sp = synthetic_ids()
    te = sp == "test"
    y2 = y.copy()
    y2[te] = np.random.default_rng(5).permutation(y[te])
    m3c = make_m3(tmp_path / "c", labels_override=y2)
    F = np.load(run1["deep"])["F_visual_deep"].copy()
    F[te] = np.random.default_rng(9).normal(size=F[te].shape)
    rvc.main(argv(m3c, save_deep(tmp_path / "d.npz", m3c, F_visual_deep=F.astype(np.float32), label=y2),
                  tmp_path / "o"))
    v = lambda d: [r for r in no_time(read_csv(d / "validation_results.csv")) if r["representation"] != "A"]
    assert v(tmp_path / "o") == v(run1["out"])
    sel = lambda d: json.loads((d / "selected_configs.json").read_text())["selected_representation"]
    assert sel(tmp_path / "o") == sel(run1["out"])


def test_a_reuses_formal_svm(m3, tmp_path, monkeypatch):
    """A uses the existing AUG-1 SVM: never refitted (only its 4 OOF fold copies), archive reproduced."""
    fitted = []
    real_fit = sklearn.svm.SVC.fit
    monkeypatch.setattr(sklearn.svm.SVC, "fit", lambda self, *a, **k: fitted.append(1) or real_fit(self, *a, **k))
    model_file = m3["aug"] / "selected_svm.joblib"
    before = hashlib.sha256(model_file.read_bytes()).hexdigest()
    rvc.main(argv(m3, save_deep(tmp_path / "d.npz", m3), tmp_path / "out"))
    assert len(fitted) == rvc.OOF_N_SPLITS
    assert hashlib.sha256(model_file.read_bytes()).hexdigest() == before
    archived = json.loads((m3["aug"] / "test_metrics.json").read_text())["test"]
    result = json.loads((tmp_path / "out" / "test_results.json").read_text())["results"]["A"]
    assert result["macro_f1"] == archived["macro_f1"] and result["confusion_matrix"] == archived["confusion_matrix"]
    z = np.load(tmp_path / "out" / "visual_representation_outputs.npz")
    np.testing.assert_array_equal(z["P_visual_handcrafted"], np.load(m3["hand"])["P_visual_handcrafted"])


# ===================================================== C. actors, AUG-1, OOF

REAL_SPLIT_ACTORS = {"train": [1, 4, 5, 6, 7, 10, 11, 12, 13, 15, 16, 18, 19, 20, 23, 24],
                     "val": [9, 14, 21, 22], "test": [2, 3, 8, 17]}  # actor_splits.csv


def test_actor_split_and_gender():
    """Actor parsing, 16/4/4 disjoint actors and 8/8, 2/2, 2/2 gender balance (counted per actor)."""
    np.testing.assert_array_equal(rvc.extract_actor_ids(["01-01-01-01-01-01", "01-08-02-02-02-24"]), [1, 24])
    with pytest.raises(rvc.DataError):
        rvc.extract_actor_ids(["01-01-01-01-01-25"])
    actor = np.array([a for s in REAL_SPLIT_ACTORS for a in REAL_SPLIT_ACTORS[s] for _ in range(3)])
    split = np.array([s for s in REAL_SPLIT_ACTORS for a in REAL_SPLIT_ACTORS[s] for _ in range(3)])
    sets = rvc.check_actor_splits(actor, split)
    assert {k: len(v) for k, v in sets.items()} == rvc.EXPECTED_ACTORS
    assert rvc.check_actor_gender_balance(actor, split) == rvc.EXPECTED_GENDERS
    swapped = np.where(actor == 1, 14, np.where(actor == 14, 1, actor))  # male 1 <-> female 14
    with pytest.raises(rvc.DataError, match="gender balance"):
        rvc.check_actor_gender_balance(swapped, split)
    with pytest.raises(rvc.DataError, match="both train and val"):
        rvc.check_actor_splits(np.array([1, 1, 2]), np.array(["train", "val", "test"]), None)


def test_actor_grouped_oof_folds(m3):
    """GroupKFold(4) over train actors; each train clip gets one OOF prediction; val/test never trained on."""
    h, _, fold, folds = oof_inputs(m3)
    tr = h["split"] == "train"
    assert set(fold[tr]) == {0, 1, 2, 3} and set(fold[~tr]) == {-1}
    assert sorted(a for f in folds for a in f["holdout_actors"]) == sorted(ACTORS["train"])
    assert all(not set(f["fit_actors"]) & set(f["holdout_actors"]) for f in folds)
    mlp = lambda: rvc.make_mlp((64,), 1e-3, 0)
    P, _, base = rvc.generate_oof_probabilities(mlp, m3["F"], h["label"], fold, "B")
    assert P.shape == (64, 8) and np.allclose(P.sum(axis=1), 1, atol=1e-5)
    X2 = m3["F"].copy()
    X2[~tr] = -99.0  # val / test rows changed
    _, _, changed = rvc.generate_oof_probabilities(mlp, X2, h["label"], fold, "B")
    probe = np.random.default_rng(3).normal(size=(5, H))
    np.testing.assert_array_equal(base[0].predict_proba(probe), changed[0].predict_proba(probe))


def test_a_oof_aug1_isolation(m3, tmp_path):
    """AUG-1 rows are matched by source_sample_id; each fold uses only its fit actors' originals and flips."""
    h, aug, fold, folds = oof_inputs(m3)
    a_model = joblib.load(m3["aug"] / "selected_svm.joblib")
    P, info, models = rvc.generate_augmented_svm_oof_probabilities(a_model, h["F"], h["label"], fold,
                                                                    h["sample_id"], aug)
    for f, i in zip(folds, info):
        ids = [s.replace("__aug01", "") for s in i["fit_augmented_ids"]]
        assert set(rvc.extract_actor_ids(ids).tolist()) == set(f["fit_actors"])
        assert i["fit_augmented_samples"] == i["fit_original_samples"] == 48
    # holdout actors' flipped features do not reach that fold's model
    aug2 = {**aug, "F": aug["F"].copy()}
    aug2["F"][np.isin(aug["actor_id"], folds[0]["holdout_actors"])] += 100.0
    _, _, models2 = rvc.generate_augmented_svm_oof_probabilities(a_model, h["F"], h["label"], fold,
                                                                 h["sample_id"], aug2)
    probe = np.random.default_rng(1).normal(size=(4, H))
    np.testing.assert_array_equal(models[0].predict_proba(probe), models2[0].predict_proba(probe))
    # shuffled cache rows give identical OOF probabilities
    z = dict(np.load(m3["aug"] / "features" / "AUG-1.npz"))
    perm = np.random.default_rng(2).permutation(64)
    z = {k: (v[perm] if np.ndim(v) >= 1 and len(v) == 64 else v) for k, v in z.items()}
    np.savez(tmp_path / "shuffled.npz", **z)
    aug3 = rvc.load_aug1_features(tmp_path / "shuffled.npz", h, 0)
    P3, _, _ = rvc.generate_augmented_svm_oof_probabilities(a_model, h["F"], h["label"], fold,
                                                            h["sample_id"], aug3)
    np.testing.assert_array_equal(P, P3)


def _bad_aug1(m3, case):
    tr = m3["split"] == "train"
    ids, y, F = m3["ids"][tr], m3["y"][tr], m3["F_aug"].copy()
    if case == "non_finite":
        F[0, 0] = np.nan
    over = {"missing_field": {}, "not_pure_flip": dict(param_rotation_deg=np.full(len(ids), 3.0)),
            "source_mismatch": dict(source_sample_id=np.array(list(ids[:-1]) + ["01-01-01-01-01-99"])),
            "non_finite": {}, "crop_grid": dict(inner_crop=224)}[case]
    return ids, y, F, over


@pytest.mark.parametrize("case,match", [("missing_field", "missing fields"),
                                        ("not_pure_flip", "pure horizontal flip"),
                                        ("source_mismatch", "missing"), ("non_finite", "NaN"),
                                        ("crop_grid", "inner_crop")])
def test_aug1_cache_errors(m3, tmp_path, case, match):
    """Typical AUG-1 cache problems are rejected."""
    ids, y, F, over = _bad_aug1(m3, case)
    p = save_aug1(tmp_path / "a.npz", ids, y, F, **over)
    if case == "missing_field":
        z = dict(np.load(p))
        z.pop("param_flip")
        np.savez(p, **z)
    with pytest.raises(rvc.DataError, match=match):
        rvc.load_aug1_features(p, hand(m3), 0)


def test_fusion_safe_probabilities(m3, run1):
    """train *_fusion rows are OOF, val/test rows the final locked model; P_visual_fusion = selected one."""
    z = np.load(run1["out"] / "visual_representation_outputs.npz")
    tr, n = m3["split"] == "train", len(m3["ids"])
    for r, final in rvc.FINAL_FIELDS.items():
        fusion = z[rvc.FUSION_FIELDS[r]]
        assert fusion.shape == (n, 8) and np.allclose(fusion.sum(axis=1), 1, atol=1e-4)
        np.testing.assert_array_equal(fusion[~tr], z[final][~tr])
        assert not np.array_equal(fusion[tr], z[final][tr])
    key = {"A": "handcrafted", "B": "deep_mlp", "A+B": "combined"}[str(z["selected_representation"])]
    np.testing.assert_array_equal(z["P_visual_fusion"], z[f"P_visual_{key}_fusion"])
    pp = json.loads((run1["out"] / "probability_protocol.json").read_text())
    assert pp["oof_used_for_selection"] is False and pp["train_probability_protocol"] == "actor_grouped_4fold_oof"


# ===================================================== D. evaluation outputs

def pred_rows(out, tag):
    return read_csv(out / f"test_predictions_{tag}.csv")


def test_metrics():
    """n_samples, Accuracy, Macro-F1 and Weighted-F1 on a small known example."""
    from sklearn.metrics import f1_score
    y = np.array([0] * 8 + [1] * 2 + [2] * 2)
    p = np.array([0] * 8 + [0, 1] + [2, 0])
    m = rvc.metric_summary(y, p)
    assert m["n_samples"] == 12 and m["accuracy"] == pytest.approx(10 / 12)
    assert m["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, p, average="weighted"))
    assert m["weighted_f1"] != pytest.approx(m["macro_f1"])


def test_prediction_csv(m3, run1):
    """Three test prediction CSVs: test rows only, same order, final-model probabilities, A uses SVM predict()."""
    z = np.load(run1["out"] / "visual_representation_outputs.npz")
    te = m3["split"] == "test"
    final = {"handcrafted": "P_visual_handcrafted", "deep": "P_visual_deep_mlp", "combined": "P_visual_combined"}
    for t in TAGS:
        rows = pred_rows(run1["out"], t)
        assert [r["sample_id"] for r in rows] == list(m3["ids"][te]) and {r["split"] for r in rows} == {"test"}
        assert [c for c in rows[0] if c.startswith("prob_")] == PROB_COLS
        P = np.array([[float(r[c]) for c in PROB_COLS] for r in rows])
        np.testing.assert_allclose(P.sum(axis=1), 1, atol=1e-4)
        np.testing.assert_allclose(P, z[final[t]][te], atol=1e-6)
        for r, p in zip(rows, P):
            k = int(r["predicted_label"])
            assert EMOTIONS[k] == r["predicted_class"]
            assert (r["correct"] == "True") == (r["true_label"] == r["predicted_label"])
            assert float(r["confidence"]) == pytest.approx(p[k])
            assert int(r["argmax_label"]) == int(p.argmax())
            assert float(r["argmax_confidence"]) == pytest.approx(p.max())
            if t != "handcrafted":
                assert k == int(p.argmax())
    a_model = joblib.load(m3["aug"] / "selected_svm.joblib")
    assert [int(r["predicted_label"]) for r in pred_rows(run1["out"], "handcrafted")] == \
        a_model.predict(m3["F"][te]).tolist()


def test_report_and_confusion_matrix(run1):
    """Classification reports match test_results.json; confusion matrices are rebuilt from the predictions."""
    tr = json.loads((run1["out"] / "test_results.json").read_text())
    for r in rvc.REPRESENTATIONS:
        tag = rvc.FILE_TAG[r]
        rep = read_csv(run1["out"] / f"classification_report_{tag}.csv")
        assert [x["class_name"] for x in rep[:8]] == EMOTIONS
        assert [x["row_type"] for x in rep[8:]] == ["accuracy", "macro_avg", "weighted_avg"]
        assert sum(int(x["support"]) for x in rep[:8]) == 16
        res = tr["results"][r]
        assert [float(x["f1_score"]) for x in rep[8:]] == pytest.approx(
            [res["accuracy"], res["macro_f1"], res["weighted_f1"]])
        cm_rows = list(csv.reader(open(run1["out"] / f"confusion_matrix_{tag}.csv")))
        cm = np.array([[int(v) for v in row[1:]] for row in cm_rows[1:]])
        rebuilt = np.zeros((8, 8), int)
        for x in pred_rows(run1["out"], tag):
            rebuilt[int(x["true_label"]), int(x["predicted_label"])] += 1
        assert cm.shape == (8, 8)
        np.testing.assert_array_equal(cm, rebuilt)


def test_png_and_comparison_table(run1):
    """Three confusion-matrix PNGs exist; the comparison table has the three configurations."""
    for t in TAGS:
        assert (run1["out"] / f"confusion_matrix_{t}.png").stat().st_size > 0
    rows = read_csv(run1["out"] / "representation_comparison.csv")
    assert [r["representation"] for r in rows] == list(TAGS)
    assert [r["classifier"] for r in rows] == ["AUG-1 DWT-SVM", "MLP", "MLP"]
    assert rows[0]["selected_hidden_layers"] == rows[0]["selected_alpha"] == ""
    for k in ("validation_accuracy", "validation_macro_f1", "validation_weighted_f1",
              "test_accuracy", "test_macro_f1", "test_weighted_f1"):
        assert all(r[k] for r in rows)


def test_final_npz(m3, run1, tmp_path):
    """NPZ order, dimensions, both probability kinds, selected outputs; optional M4 fields never faked."""
    z = np.load(run1["out"] / "visual_representation_outputs.npz")
    n = len(m3["ids"])
    np.testing.assert_array_equal(z["sample_id"], m3["ids"])
    np.testing.assert_array_equal(z["split"], m3["split"])
    assert list(z["class_names"]) == EMOTIONS
    assert (z["F_visual_handcrafted"].shape, z["F_visual_deep"].shape, z["F_visual_combined"].shape) == \
        ((n, H), (n, D), (n, H + D))
    for k in (*rvc.FINAL_FIELDS.values(), *rvc.FUSION_FIELDS.values(), "P_visual", "P_visual_fusion"):
        assert z[k].shape == (n, 8)
    key = {"A": "handcrafted", "B": "deep", "A+B": "combined"}[str(z["selected_representation"])]
    np.testing.assert_array_equal(z["F_visual"], z[f"F_visual_{key}"])
    assert "F_visual_deep_sequence" in z.files and "P_visual_deep_source" in z.files
    deep = save_deep(tmp_path / "d.npz", m3, drop=("F_visual_sequence", "P_visual_deep"))
    rvc.main(argv(m3, deep, tmp_path / "o"))
    z2 = np.load(tmp_path / "o" / "visual_representation_outputs.npz")
    assert "F_visual_deep_sequence" not in z2.files and "P_visual_deep_source" not in z2.files


def test_reproducible_and_output_safety(m3, run1, tmp_path):
    """Same input and seed give the same choices and probabilities; existing results and protected folders are not overwritten."""
    rvc.main(argv(m3, run1["deep"], tmp_path / "again"))
    assert no_time(read_csv(tmp_path / "again" / "validation_results.csv")) == \
        no_time(read_csv(run1["out"] / "validation_results.csv"))
    a = np.load(run1["out"] / "visual_representation_outputs.npz")
    b = np.load(tmp_path / "again" / "visual_representation_outputs.npz")
    assert str(a["selected_representation"]) == str(b["selected_representation"])
    for k in ("P_visual", "P_visual_fusion", "P_visual_deep_mlp", "P_visual_combined"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-6)
    with pytest.raises(SystemExit, match="overwrite"):
        rvc.main(argv(m3, run1["deep"], run1["out"]))
    with pytest.raises(SystemExit, match="Refusing"):
        rvc.main(argv(m3, run1["deep"], ROOT / "outputs_augmentation"))
