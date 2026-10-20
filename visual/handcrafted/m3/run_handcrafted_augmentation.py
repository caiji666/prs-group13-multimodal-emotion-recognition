"""Train-only data augmentation experiment for the M3 handcrafted DWT-SVM (fixed config D).

Configuration D is fixed (inner_crop=180, grid=4, 320 features / frame, temporal
mean pooling); only the SVM training data changes:

    AUG-0  original train only (960)                       -> must reproduce the D baseline
    AUG-1  original train + horizontally flipped copy (1920)
    AUG-2  original train + one light combined augmented copy (1920)

What this measures: whether augmenting the TRAINING data makes the DWT-SVM
classifier more robust. The handcrafted representation itself
(F_visual_handcrafted) is always the DWT feature of the ORIGINAL clip; the
augmentation only changes the SVM training set and therefore the exported
P_visual_handcrafted. The augmented clips have no M4 deep features, so they
cannot be used for Handcrafted + Deep per-sample concatenation; fusion must
still align on the original sample_id. Using augmented clips in a combined MLP
would require M4 to export deep features for exactly the same augmented clips.

Protocol (identical for the three configurations):
    - same faces, actor split, sample_id order, label order, seed, SVM search space;
    - validation and test always use the original D features (never augmented);
    - StandardScaler is fitted inside the SVM pipeline on that configuration's
      training set (original + augmented train);
    - SVM hyperparameters and the augmentation configuration are chosen on
      validation Macro-F1 only (ties: AUG-0 > AUG-1 > AUG-2, simplest first);
    - the test set is evaluated once, for the selected configuration only.

Inputs (read only):
    --faces-dir           data/faces/*.npz shared 224x224 face clips
    --baseline-features   outputs_ablation/features/D.npz (pooled D features, AUG-0 and val/test)
    --sequence-features   data/dwt_features.npz (D per-frame features + timing, for the export)

Outputs (--out-dir, default outputs_augmentation/):
    augmentation_validation_results.csv / .json, selected_augmentation.json,
    test_metrics.json, confusion_matrix.csv, selected_svm.joblib,
    augmentation_config.json, features/AUG-1.npz, features/AUG-2.npz (train only),
    m3_visual_handcrafted_augmented.npz (original 1440 samples only)

Usage:
    uv run python -m m3.run_handcrafted_augmentation --faces-dir data/faces --baseline-features outputs_ablation/features/D.npz --out-dir outputs_augmentation --workers 4 --seed 0
Smoke test (separate folder):
    uv run python -m m3.run_handcrafted_augmentation --out-dir outputs_augmentation_smoke --workers 2 --max-per-class 1
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from m3 import run_handcrafted_ablation as abl
from m3.augment_faces import (AUG2_RANGES, AUG_CONFIGS, PARAM_KEYS, TRAINING_AUGMENTATION,
                              augment_clip, check_clip, check_params, sample_params)
from m3.common import EMOTIONS, ROOT, load_actor_splits
from m3.dwt_features import (N_FRAMES, clip_features, face_files, feature_names,
                             load_face_file, validate_config)
from m3.train_svm import GRID, TIMING_KEYS, scores

D_CROP, D_GRID = abl.CONFIGS["D"]
D_DIM = len(feature_names(D_GRID))
AUG_SUFFIX = "__aug01"
FORMAL_DIR = "outputs_augmentation"
PROTECTED_DIRS = ("outputs", "outputs_v1_global20", "outputs_ablation", "data")
TIE_RULE = "highest validation Macro-F1; exact ties -> AUG-0, then AUG-1, then AUG-2 (simplest first)"
CSV_COLUMNS = [
    "config_name", "original_train_samples", "augmented_train_samples", "total_train_samples",
    "validation_samples", "feature_dim", "seed", "best_C", "best_gamma", "best_class_weight",
    "train_macro_f1", "train_accuracy", "validation_macro_f1", "validation_accuracy",
    "feature_seconds", "svm_seconds", "runtime_seconds",
]


# ------------------------------------------------------------- baseline inputs

def load_baseline(path, sample_ids, labels, splits):
    """Pooled D features for all samples; every check failure raises (no silent fallback)."""
    z = np.load(path)

    def fail(msg):
        raise ValueError(f"baseline features {path}: {msg}")

    for k in ("sample_id", "split", "label", "F_pooled", "inner_crop", "grid"):
        if k not in z.files:
            fail(f"missing field {k!r}")
    ids = list(z["sample_id"])
    if len(ids) != len(sample_ids):
        fail(f"{len(ids)} samples, faces dir has {len(sample_ids)}")
    if len(set(ids)) != len(ids):
        fail("duplicate sample_id")
    if ids != list(sample_ids):
        fail("sample_id order differs from the faces dir")
    if not np.array_equal(z["split"], splits):
        fail("split differs from actor_splits.csv")
    if not np.array_equal(z["label"], labels):
        fail("label differs from the sample_id emotion code")
    if (int(z["inner_crop"]), int(z["grid"])) != (D_CROP, D_GRID):
        fail(f"not config D (inner_crop={int(z['inner_crop'])}, grid={int(z['grid'])})")
    X = z["F_pooled"]
    if X.shape != (len(ids), D_DIM):
        fail(f"F_pooled shape {X.shape}, expected {(len(ids), D_DIM)}")
    if not np.isfinite(X).all():
        fail("NaN or Inf in F_pooled")
    return X.astype(np.float32)


def load_sequences(path, sample_ids, F_pooled):
    """Per-frame D features + timing fields of the original samples (for the export)."""
    z = dict(np.load(path))

    def fail(msg):
        raise ValueError(f"sequence features {path}: {msg}")

    if list(z.get("sample_id", [])) != list(sample_ids):
        fail("sample_id order differs from the faces dir")
    if z["F_sequence"].shape != (len(sample_ids), N_FRAMES, D_DIM):
        fail(f"F_sequence shape {z['F_sequence'].shape}")
    if list(z["feature_names"]) != feature_names(D_GRID):
        fail("feature_names are not the D (grid=4) names")
    missing = [k for k in TIMING_KEYS if k not in z]
    if missing:
        fail(f"missing timing fields {missing}")
    if not np.allclose(z["F_sequence"].mean(axis=1), F_pooled, rtol=0, atol=1e-6):
        fail("temporal mean of F_sequence does not match the baseline F_pooled")
    return z


# ------------------------------------------------------- augmented train features

def _augmented_features(job):
    """Worker: one train clip -> pooled D features of each augmented copy."""
    path, sid, configs, seed = job
    frames = load_face_file(path)["frames"]
    check_clip(frames)
    out = {}
    for cfg in configs:
        t = time.perf_counter()
        params = sample_params(cfg, sid, seed)  # depends on (cfg, sid, seed) only
        clip = augment_clip(frames, params)
        out[cfg] = (clip_features(clip, D_GRID, D_CROP).mean(axis=0), params,
                    time.perf_counter() - t)
    return sid, out


def extract_augmented(items, configs, seed, workers=1):
    """items: [(sample_id, face path)] in any order -> {cfg: {sid: (features, params)}}, seconds."""
    jobs = [(path, sid, tuple(configs), seed) for sid, path in items]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(tqdm(pool.map(_augmented_features, jobs, chunksize=8), total=len(jobs)))
    else:
        results = [_augmented_features(j) for j in tqdm(jobs)]
    feats = {c: {} for c in configs}
    secs = {c: 0.0 for c in configs}
    for sid, out in results:
        for c in configs:
            f, p, s = out[c]
            feats[c][sid] = (f.astype(np.float32), p)
            secs[c] += s
    return feats, secs


def params_arrays(params_list):
    return {k: np.array([p[k] for p in params_list]) for k in PARAM_KEYS}


def save_aug_cache(path, cfg, seed, train_ids, y_tr, X_aug, params_list, seconds):
    np.savez(path, source_sample_id=np.array(train_ids),
             augmented_sample_id=np.array([s + AUG_SUFFIX for s in train_ids]),
             label=y_tr, split=np.full(len(train_ids), "train"), F_pooled=X_aug,
             config_name=cfg, seed=seed, feature_dim=D_DIM, inner_crop=D_CROP, grid=D_GRID,
             augmentation_ranges=json.dumps(AUG2_RANGES),
             params_json=np.array([json.dumps(p, sort_keys=True) for p in params_list]),
             feature_seconds=seconds, **{f"param_{k}": v for k, v in params_arrays(params_list).items()})


def load_aug_cache(path, cfg, seed, train_ids, y_tr):
    """Load a cache only if every field matches; otherwise raise."""
    z = np.load(path)

    def fail(msg):
        raise ValueError(f"augmentation cache {path}: {msg}; delete it or run without --reuse-features")

    if str(z["config_name"]) != cfg:
        fail(f"config_name {z['config_name']} != {cfg}")
    if int(z["seed"]) != seed:
        fail(f"seed {int(z['seed'])} != {seed}")
    if list(z["source_sample_id"]) != list(train_ids):
        fail("source_sample_id set or order differs")
    if list(z["augmented_sample_id"]) != [s + AUG_SUFFIX for s in train_ids]:
        fail("augmented_sample_id differs")
    if not np.array_equal(z["label"], y_tr):
        fail("label differs")
    if set(z["split"]) != {"train"}:
        fail("cache contains non-train samples")
    if (int(z["inner_crop"]), int(z["grid"]), int(z["feature_dim"])) != (D_CROP, D_GRID, D_DIM):
        fail("not config D")
    if z["F_pooled"].shape != (len(train_ids), D_DIM) or not np.isfinite(z["F_pooled"]).all():
        fail(f"bad F_pooled {z['F_pooled'].shape}")
    if json.loads(str(z["augmentation_ranges"])) != json.loads(json.dumps(AUG2_RANGES)):
        fail("augmentation ranges changed")
    cached = [json.loads(s) for s in z["params_json"]]
    for sid, p in zip(train_ids, cached):
        check_params(p)
        if p != sample_params(cfg, sid, seed):
            fail(f"parameters of {sid} differ from the current derivation")
    return z["F_pooled"].astype(np.float32), cached, float(z["feature_seconds"])


def augmented_train_features(train_items, y_tr, configs, seed, out_dir, workers, reuse):
    """{cfg: (X_aug [n_train, 320], params list, seconds)} in train_items order."""
    feat_dir = out_dir / "features"
    train_ids = [sid for sid, _ in train_items]
    result, todo = {}, []
    for cfg in configs:
        cache = feat_dir / f"{cfg}.npz"
        if reuse and cache.exists():
            result[cfg] = load_aug_cache(cache, cfg, seed, train_ids, y_tr)
            print(f"[{cfg}] reusing {cache}")
        else:
            todo.append(cfg)
    if todo:
        print(f"Augmenting {len(train_items)} train clips for {todo}")
        feats, secs = extract_augmented(train_items, todo, seed, workers)
        feat_dir.mkdir(parents=True, exist_ok=True)
        for cfg in todo:
            X = np.stack([feats[cfg][s][0] for s in train_ids])
            params = [feats[cfg][s][1] for s in train_ids]
            save_aug_cache(feat_dir / f"{cfg}.npz", cfg, seed, train_ids, y_tr, X, params, secs[cfg])
            result[cfg] = (X, params, secs[cfg])
    return result


# ------------------------------------------------------------------ evaluation

def run_validation_stage(X_tr, y_tr, X_va, y_va, aug, seed):
    """Validation-only comparison; the test split is not an argument.

    aug: {cfg: (X_aug, params, seconds)} for AUG-1 / AUG-2.
    Returns rows (AUG-0, AUG-1, AUG-2 order) and fitted models.
    """
    rows, models = [], {}
    for cfg in AUG_CONFIGS:
        if cfg == "AUG-0":
            Xt, yt, n_aug, fsec = X_tr, y_tr, 0, 0.0
        else:
            X_aug, _, fsec = aug[cfg]
            Xt, yt, n_aug = np.vstack([X_tr, X_aug]), np.concatenate([y_tr, y_tr]), len(X_aug)
        res, models[cfg] = abl.evaluate_config(Xt, yt, X_va, y_va, seed)
        row = {"config_name": cfg, "original_train_samples": len(X_tr),
               "augmented_train_samples": n_aug, "total_train_samples": len(Xt),
               "validation_samples": len(X_va), "feature_dim": Xt.shape[1], "seed": seed,
               **res, "feature_seconds": fsec}
        row["runtime_seconds"] = row["feature_seconds"] + row["svm_seconds"]
        rows.append(row)
        print(f"[{cfg}] train={len(Xt)}  C={res['best_C']} gamma={res['best_gamma']} "
              f"cw={res['best_class_weight']}  val Macro-F1={res['validation_macro_f1']:.4f} "
              f"acc={res['validation_accuracy']:.4f}")
    return rows, models


def evaluate_test_once(model, X_te, y_te):
    return scores(y_te, model.predict(X_te))


def export_candidate(path, model, seq, X_all, labels, splits, sample_ids, cfg, seed, aug_cfg):
    """Original samples only; P from the SVM trained with the selected scheme."""
    if any(AUG_SUFFIX in s for s in sample_ids):
        raise ValueError("augmented sample_id in the export")
    if list(model.classes_) != list(range(len(EMOTIONS))):
        raise ValueError(f"model.classes_ {list(model.classes_)} != 0..7")
    P = model.predict_proba(X_all).astype(np.float32)  # columns = classes_ = EMOTIONS order
    if P.shape != (len(sample_ids), len(EMOTIONS)):
        raise ValueError(f"P shape {P.shape}")
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("probability rows do not sum to 1")
    n, t = seq["F_sequence"].shape[:2]
    np.savez(
        path,
        sample_id=np.array(sample_ids), split=splits, label=labels,
        F_sequence=seq["F_sequence"].astype(np.float32),
        sequence_length=np.full(n, t, dtype=np.int64),
        padding_mask=np.zeros((n, t), dtype=bool),
        F_visual_handcrafted=X_all.astype(np.float32),
        P_visual_handcrafted=P,
        class_names=np.array(EMOTIONS),
        feature_names=seq["feature_names"],
        **{k: seq[k] for k in TIMING_KEYS},
        training_augmentation=TRAINING_AUGMENTATION[cfg],
        augmentation_seed=seed,
        augmentation_config=json.dumps(aug_cfg, default=str),
    )
    return P


# ----------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--faces-dir", default="data/faces")
    ap.add_argument("--baseline-features", default="outputs_ablation/features/D.npz")
    ap.add_argument("--sequence-features", default="data/dwt_features.npz")
    ap.add_argument("--out-dir", default=FORMAL_DIR)
    ap.add_argument("--actor-splits", default=str(ROOT / "actor_splits.csv"))
    ap.add_argument("--seed", type=int, default=0, help="global seed (augmentation + SVC)")
    ap.add_argument("--workers", type=int, default=1, help="processes; does not change results")
    ap.add_argument("--expected-samples", type=int, default=1440)
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="smoke test: first N samples of every split x emotion (separate --out-dir)")
    ap.add_argument("--reuse-features", action="store_true",
                    help="load features/AUG-*.npz from --out-dir after checking them")
    ap.add_argument("--overwrite", action="store_true", help="replace results in --out-dir")
    args = ap.parse_args(argv)

    validate_config(D_CROP, D_GRID)
    out_dir = Path(args.out_dir)
    smoke = args.max_per_class is not None
    if smoke and args.max_per_class < 1:
        raise SystemExit("--max-per-class must be >= 1")
    if smoke and out_dir.resolve() == (ROOT / FORMAL_DIR).resolve():
        raise SystemExit(f"Smoke runs must not write into {FORMAL_DIR}/; use another --out-dir")
    abl.check_out_dir(out_dir, args.faces_dir, args.overwrite, protected=PROTECTED_DIRS,
                      marker="selected_augmentation.json")

    files = face_files(args.faces_dir)
    sample_ids = [f.stem for f in files]
    if len(sample_ids) != args.expected_samples:
        raise SystemExit(f"{args.faces_dir} has {len(sample_ids)} face files, "
                         f"expected {args.expected_samples}")
    labels, splits = abl.build_metadata(sample_ids, load_actor_splits(args.actor_splits))
    X_all = load_baseline(args.baseline_features, sample_ids, labels, splits)
    seq = load_sequences(args.sequence_features, sample_ids, X_all)
    if smoke:
        keep = abl.select_subset(sample_ids, labels, splits, args.max_per_class)
        files = [files[i] for i in keep]
        sample_ids = [sample_ids[i] for i in keep]
        labels, splits, X_all = labels[keep], splits[keep], X_all[keep]
        seq = {k: (v[keep] if k in ("F_sequence", *TIMING_KEYS) else v) for k, v in seq.items()}
    abl.check_aligned(sample_ids, labels, splits, X_all)
    tr, va, te = (splits == s for s in abl.SPLITS)
    counts = {s: int((splits == s).sum()) for s in abl.SPLITS}
    print(f"Samples per split: {counts}" + (" (smoke subset)" if smoke else ""))

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_items = [(sample_ids[i], files[i]) for i in np.flatnonzero(tr)]
    aug = augmented_train_features(train_items, labels[tr], AUG_CONFIGS[1:], args.seed,
                                   out_dir, args.workers, args.reuse_features)

    # Stage 1: validation only (original val features). Stage 2: test once, selected config.
    rows, models = run_validation_stage(X_all[tr], labels[tr], X_all[va], labels[va], aug, args.seed)
    chosen = abl.select_config(rows)
    chosen_row = next(r for r in rows if r["config_name"] == chosen)
    test = evaluate_test_once(models[chosen], X_all[te], labels[te])
    print(f"Selected {chosen} (val Macro-F1={chosen_row['validation_macro_f1']:.4f}); "
          f"TEST Macro-F1={test['macro_f1']:.4f} acc={test['accuracy']:.4f}")

    aug_cfg = {
        "configs": {
            "AUG-0": "original train only",
            "AUG-1": "original train + horizontally flipped copy of every train clip",
            "AUG-2": "original train + one light combined augmented copy of every train clip",
        },
        "aug2_ranges": AUG2_RANGES,
        "clip_consistent": "one parameter set per clip, applied to all 16 frames",
        "per_clip_seed": "sha256('<config>|<sample_id>|<seed>')[:8] little-endian -> numpy default_rng",
        "geometric_border": "cv2.BORDER_REFLECT_101, bilinear",
        "photometric": "x' = ((x - clip_mean) * contrast + clip_mean) * brightness, clipped to 0..255",
        "augmented_splits": ["train"],
        "validation_and_test": "original D features, never augmented",
        "feature_pipeline": "augmented 224x224 RGB -> D (crop 180, Haar DWT, 4x4 grid, 5 stats, "
                            "320 / frame) -> mean over 16 frames",
        "seed": args.seed,
        "limitations": "augmented clips have no M4 deep features; they only train the DWT-SVM and "
                       "cannot be used for Handcrafted + Deep per-sample concatenation",
    }
    protocol = {
        "seed": args.seed, "svm_search_space": GRID, "fixed_config": "D",
        "inner_crop": D_CROP, "grid": D_GRID,
        "scaler": "StandardScaler fitted on each configuration's training set only (SVM pipeline)",
        "hyperparameter_selection": "validation Macro-F1 (m3.train_svm.select_hyperparams)",
        "config_selection": TIE_RULE,
        "test_set": "evaluated once, for the selected configuration only",
        "faces_dir": str(args.faces_dir), "baseline_features": str(args.baseline_features),
        "sequence_features": str(args.sequence_features),
        "counts": counts, "smoke_subset_max_per_class": args.max_per_class,
    }
    with open(out_dir / "augmentation_validation_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "augmentation_validation_results.json", "w") as fh:
        json.dump({"protocol": protocol, "results": rows}, fh, indent=2, default=str)
    with open(out_dir / "augmentation_config.json", "w") as fh:
        json.dump(aug_cfg, fh, indent=2, default=str)
    with open(out_dir / "selected_augmentation.json", "w") as fh:
        json.dump({"selected_config": chosen_row, "tie_breaking": TIE_RULE,
                   "final_svm_params": {"C": chosen_row["best_C"], "gamma": chosen_row["best_gamma"],
                                        "class_weight": chosen_row["best_class_weight"]},
                   "svm_search_space": GRID, "protocol": protocol}, fh, indent=2, default=str)
    with open(out_dir / "test_metrics.json", "w") as fh:
        json.dump({"config_name": chosen, "training_augmentation": TRAINING_AUGMENTATION[chosen],
                   "best_params": {"C": chosen_row["best_C"], "gamma": chosen_row["best_gamma"],
                                   "class_weight": chosen_row["best_class_weight"]},
                   "n_test": counts["test"], "test": test,
                   "smoke_subset_max_per_class": args.max_per_class}, fh, indent=2, default=str)
    with open(out_dir / "confusion_matrix.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred", *EMOTIONS])
        for name, row in zip(EMOTIONS, test["confusion_matrix"]):
            w.writerow([name, *row])
    joblib.dump(models[chosen], out_dir / "selected_svm.joblib")
    export_candidate(out_dir / "m3_visual_handcrafted_augmented.npz", models[chosen], seq, X_all,
                     labels, splits, sample_ids, chosen, args.seed, aug_cfg)
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return rows, chosen, test


if __name__ == "__main__":
    main()
