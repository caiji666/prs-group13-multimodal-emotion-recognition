"""Controlled ablation of the handcrafted DWT features: inner crop x local grid.

v1 -> v2 changed two things at once (inner 180x180 crop and 4x4 subband grid).
This runner separates them with four configurations that differ only in
(inner_crop, grid):

    A: inner_crop=224, grid=1   (= v1, 20 features / frame)
    B: inner_crop=180, grid=1   (20)
    C: inner_crop=224, grid=4   (320)
    D: inner_crop=180, grid=4   (= v2, 320)

Everything else is identical for the four configurations: the shared 224x224
face frames (read only, never rewritten), sample_id order, actor_splits.csv,
label order, temporal mean pooling, StandardScaler fitted on train only, the
SVM search space of m3.train_svm.GRID and the random seed.

Protocol:
    1. every configuration: SVM hyperparameters chosen on validation Macro-F1,
       train and validation scores recorded;
    2. the configuration with the highest validation Macro-F1 is selected
       (ties: first in A, B, C, D order);
    3. only the selected configuration is evaluated on the test set, once.

Outputs (in --out-dir, default outputs_ablation/):
    ablation_validation_results.csv / .json   one row per configuration (no test scores)
    selected_config.json                      chosen configuration + selection rule
    test_metrics.json, confusion_matrix.csv   test result of the chosen configuration only
    selected_svm.joblib                       StandardScaler + SVM of the chosen configuration
    features/<config>.npz                     pooled features per configuration (cache)

Usage:
    uv run python -m m3.run_handcrafted_ablation --faces-dir data/faces --out-dir outputs_ablation --workers 4
Smoke test (first 2 samples of each split x emotion):
    uv run python -m m3.run_handcrafted_ablation --max-per-class 2 --out-dir outputs_ablation_smoke
"""
import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from m3.common import (EMOTIONS, ROOT, actor_from_sample_id, label_from_sample_id,
                       load_actor_splits)
from m3.dwt_features import (FRAME_SIZE, N_FRAMES, clip_features, face_files, feature_names,
                             load_face_file, validate_config)
from m3.train_svm import GRID, make_model, scores, select_hyperparams

CONFIGS = {  # name -> (inner_crop, grid)
    "A": (224, 1),
    "B": (180, 1),
    "C": (224, 4),
    "D": (180, 4),
}
SPLITS = ("train", "val", "test")
PROTECTED_DIRS = ("outputs", "outputs_v1_global20", "data")
CSV_COLUMNS = [
    "config_name", "inner_crop", "grid", "feature_dim_per_frame", "pooled_feature_dim",
    "best_C", "best_gamma", "best_class_weight",
    "train_macro_f1", "train_accuracy", "validation_macro_f1", "validation_accuracy",
    "feature_seconds", "svm_seconds", "runtime_seconds",
]


# ---------------------------------------------------------------- data checks

def check_sample_id(sid):
    parts = sid.split("-")
    if len(parts) != 6 or any(len(p) != 2 or not p.isdigit() for p in parts) or parts[0] != "01":
        raise ValueError(f"Not a team speech sample_id: {sid!r}")


def build_metadata(sample_ids, actor_splits):
    """sample_id order -> (label, split) arrays; checks uniqueness and format."""
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("duplicate sample_id")
    for sid in sample_ids:
        check_sample_id(sid)
    labels = np.array([label_from_sample_id(s) for s in sample_ids], dtype=np.int64)
    if labels.min() < 0 or labels.max() >= len(EMOTIONS):
        raise ValueError("label outside 0..7")
    try:
        splits = np.array([actor_splits[actor_from_sample_id(s)] for s in sample_ids])
    except KeyError as e:
        raise ValueError(f"actor {e} missing from actor_splits.csv") from None
    return labels, splits


def check_aligned(sample_ids, labels, splits, X):
    n = len(sample_ids)
    if not (len(labels) == len(splits) == X.shape[0] == n):
        raise ValueError(f"row counts differ: sample_id={n}, label={len(labels)}, "
                         f"split={len(splits)}, features={X.shape[0]}")
    if len(set(sample_ids)) != n:
        raise ValueError("duplicate sample_id")
    if not np.isfinite(X).all():
        raise ValueError("non-finite feature values")
    for s in SPLITS:
        if not (splits == s).any():
            raise ValueError(f"split {s!r} is empty")


def select_subset(sample_ids, labels, splits, max_per_class):
    """Keep the first max_per_class samples (sorted sample_id order) of every split x label."""
    seen, keep = {}, []
    for i, key in enumerate(zip(splits, labels)):
        if seen.get(key, 0) < max_per_class:
            seen[key] = seen.get(key, 0) + 1
            keep.append(i)
    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------- feature extraction

def _pooled_features(args):
    """Worker: one face file -> pooled features + compute time for every configuration."""
    path, configs = args
    frames = load_face_file(path)["frames"]
    if frames.shape != (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3):
        raise ValueError(f"{path.name}: expected frames [{N_FRAMES}, {FRAME_SIZE}, {FRAME_SIZE}, 3], "
                         f"got {frames.shape}")
    pooled, secs = {}, {}
    for name, (crop, grid) in configs.items():
        t = time.perf_counter()
        pooled[name] = clip_features(frames, grid, crop).mean(axis=0)  # temporal mean pooling
        secs[name] = time.perf_counter() - t
    return path.stem, pooled, secs


def extract_features(files, configs, workers=1):
    """Each face file is read once; returns {config: [N, D]}, {config: summed seconds}."""
    jobs = [(f, configs) for f in files]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            results = list(tqdm(pool.map(_pooled_features, jobs, chunksize=8), total=len(jobs)))
    else:
        results = [_pooled_features(j) for j in tqdm(jobs)]
    if [r[0] for r in results] != [f.stem for f in files]:
        raise RuntimeError("feature rows are not in sample_id order")
    X = {n: np.stack([r[1][n] for r in results]).astype(np.float32) for n in configs}
    secs = {n: float(sum(r[2][n] for r in results)) for n in configs}
    return X, secs


def load_or_extract(files, sample_ids, labels, splits, out_dir, workers, reuse):
    feat_dir = out_dir / "features"
    cache = {n: feat_dir / f"{n}.npz" for n in CONFIGS}
    todo = {}
    X, secs = {}, {}
    for name, cfg in CONFIGS.items():
        if reuse and cache[name].exists():
            z = np.load(cache[name])
            if (list(z["sample_id"]) == list(sample_ids)
                    and (int(z["inner_crop"]), int(z["grid"])) == cfg):
                X[name], secs[name] = z["F_pooled"], float(z["feature_seconds"])
                print(f"[{name}] reusing {cache[name]}")
                continue
        todo[name] = cfg
    if todo:
        print(f"Extracting {sorted(todo)} from {len(files)} face files")
        new_X, new_secs = extract_features(files, todo, workers)
        feat_dir.mkdir(parents=True, exist_ok=True)
        for name, (crop, grid) in todo.items():
            np.savez(cache[name], sample_id=np.array(sample_ids), split=splits, label=labels,
                     F_pooled=new_X[name], feature_names=np.array(feature_names(grid)),
                     inner_crop=crop, grid=grid, feature_seconds=new_secs[name])
        X.update(new_X)
        secs.update(new_secs)
    return X, secs


# ------------------------------------------------------------------ evaluation

def evaluate_config(X_tr, y_tr, X_va, y_va, seed):
    """Validation-only model selection for one configuration (no test data passed in)."""
    t = time.perf_counter()
    best, _ = select_hyperparams(X_tr, y_tr, X_va, y_va, random_state=seed, verbose=False)
    model = make_model(*best, random_state=seed).fit(X_tr, y_tr)
    pred_tr, pred_va = model.predict(X_tr), model.predict(X_va)
    result = {
        "best_C": best[0], "best_gamma": best[1], "best_class_weight": best[2],
        "train_macro_f1": f1_score(y_tr, pred_tr, average="macro"),
        "train_accuracy": accuracy_score(y_tr, pred_tr),
        "validation_macro_f1": f1_score(y_va, pred_va, average="macro"),
        "validation_accuracy": accuracy_score(y_va, pred_va),
        "svm_seconds": time.perf_counter() - t,
    }
    return result, model


def select_config(rows):
    """Highest validation Macro-F1; ties keep the earlier configuration (A, B, C, D order)."""
    best = None
    for r in rows:
        if best is None or r["validation_macro_f1"] > best["validation_macro_f1"]:
            best = r
    return best["config_name"]


def run_validation_stage(X, labels, splits, feature_secs, seed):
    tr, va = splits == "train", splits == "val"
    rows, models = [], {}
    for name, (crop, grid) in CONFIGS.items():
        res, models[name] = evaluate_config(X[name][tr], labels[tr], X[name][va], labels[va], seed)
        dim = len(feature_names(grid))
        if X[name].shape[1] != dim:
            raise ValueError(f"[{name}] pooled dim {X[name].shape[1]} != {dim}")
        row = {"config_name": name, "inner_crop": crop, "grid": grid,
               "feature_dim_per_frame": dim, "pooled_feature_dim": X[name].shape[1], **res,
               "feature_seconds": feature_secs[name]}
        row["runtime_seconds"] = row["feature_seconds"] + row["svm_seconds"]
        rows.append(row)
        print(f"[{name}] crop={crop} grid={grid} dim={dim}  C={res['best_C']} "
              f"gamma={res['best_gamma']} cw={res['best_class_weight']}  "
              f"val Macro-F1={res['validation_macro_f1']:.4f} acc={res['validation_accuracy']:.4f}")
    return rows, models


# ----------------------------------------------------------------------- main

def check_out_dir(out_dir, faces_dir, overwrite, protected=PROTECTED_DIRS,
                  marker="selected_config.json"):
    """Refuse protected folders, and a folder that already holds results unless overwrite."""
    out = Path(out_dir).resolve()
    forbidden = [(ROOT / d).resolve() for d in protected] + [Path(faces_dir).resolve()]
    for p in forbidden:
        if out == p or p in out.parents:
            raise SystemExit(f"Refusing to write results into {out} (protected: {p})")
    if (out / marker).exists() and not overwrite:
        raise SystemExit(f"{out} already holds results; pass --overwrite or choose a new --out-dir")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--faces-dir", default="data/faces")
    ap.add_argument("--out-dir", default="outputs_ablation")
    ap.add_argument("--actor-splits", default=str(ROOT / "actor_splits.csv"))
    ap.add_argument("--seed", type=int, default=0, help="SVC random_state (Platt scaling)")
    ap.add_argument("--workers", type=int, default=1, help="processes for feature extraction")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="smoke test: keep the first N samples of every split x emotion")
    ap.add_argument("--reuse-features", action="store_true",
                    help="load features/<config>.npz from --out-dir if it matches")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace results already in --out-dir")
    args = ap.parse_args(argv)

    for crop, grid in CONFIGS.values():
        validate_config(crop, grid)
    if args.max_per_class is not None and args.max_per_class < 1:
        raise SystemExit("--max-per-class must be >= 1")
    out_dir = Path(args.out_dir)
    check_out_dir(out_dir, args.faces_dir, args.overwrite)

    files = face_files(args.faces_dir)
    if not files:
        raise SystemExit(f"No face files in {args.faces_dir}")
    sample_ids = [f.stem for f in files]
    labels, splits = build_metadata(sample_ids, load_actor_splits(args.actor_splits))
    if args.max_per_class is not None:
        keep = select_subset(sample_ids, labels, splits, args.max_per_class)
        files = [files[i] for i in keep]
        sample_ids, labels, splits = [sample_ids[i] for i in keep], labels[keep], splits[keep]
    counts = {s: int((splits == s).sum()) for s in SPLITS}
    print(f"Samples per split: {counts}" + ("" if args.max_per_class else " (expected 960 / 240 / 240)"))

    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    X, feature_secs = load_or_extract(files, sample_ids, labels, splits, out_dir,
                                      args.workers, args.reuse_features)
    for name in CONFIGS:
        check_aligned(sample_ids, labels, splits, X[name])

    # Stage 1: validation only. Stage 2: test once for the selected configuration.
    rows, models = run_validation_stage(X, labels, splits, feature_secs, args.seed)
    chosen = select_config(rows)
    chosen_row = next(r for r in rows if r["config_name"] == chosen)
    te = splits == "test"
    test = scores(labels[te], models[chosen].predict(X[chosen][te]))
    print(f"Selected {chosen} (val Macro-F1={chosen_row['validation_macro_f1']:.4f}); "
          f"TEST Macro-F1={test['macro_f1']:.4f} acc={test['accuracy']:.4f}")

    protocol = {
        "seed": args.seed,
        "svm_search_space": GRID,
        "scaler": "StandardScaler fitted on train split only (inside the SVM pipeline)",
        "pooling": "temporal mean over 16 frames",
        "hyperparameter_selection": "validation Macro-F1 (m3.train_svm.select_hyperparams)",
        "config_selection": "highest validation Macro-F1; ties -> first in A, B, C, D order",
        "test_set": "evaluated once, for the selected configuration only",
        "faces_dir": str(args.faces_dir),
        "base_frame_size": FRAME_SIZE,
        "class_names": EMOTIONS,
        "counts": counts,
        "smoke_subset_max_per_class": args.max_per_class,
    }
    with open(out_dir / "ablation_validation_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "ablation_validation_results.json", "w") as fh:
        json.dump({"protocol": protocol, "results": rows}, fh, indent=2, default=str)
    with open(out_dir / "selected_config.json", "w") as fh:
        json.dump({"selected_config": chosen_row, "protocol": protocol}, fh, indent=2, default=str)
    with open(out_dir / "test_metrics.json", "w") as fh:
        json.dump({"config_name": chosen, "inner_crop": chosen_row["inner_crop"],
                   "grid": chosen_row["grid"],
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
    print(f"Done in {time.perf_counter() - t0:.1f}s -> {out_dir}")
    return rows, chosen, test


if __name__ == "__main__":
    main()
