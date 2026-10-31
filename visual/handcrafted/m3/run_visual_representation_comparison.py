"""Level-2 visual representation comparison (A / B / A+B), proposal Sec. 3.

The proposal compares three COMPLETE visual configurations:

    A    handcrafted DWT feature (320)          -> RBF SVM   (existing AUG-1 model, not retrained)
    B    M4 deep feature F_visual_deep (D)      -> StandardScaler + MLPClassifier
    A+B  [handcrafted, deep] along axis 1 (320+D) -> StandardScaler + MLPClassifier

This is a configuration comparison, NOT a pure feature ablation with one fixed
classifier: A uses an SVM, B and A+B use an MLP, so differences cannot be
attributed to the features alone. A's AUG-1 SVM was trained on 960 original +
960 horizontally flipped train clips; the B and A+B MLPs use only the 960
original train clips, because augmented clips have no M4 deep features. AUG-1
augmented samples are never concatenated with M4 deep features and are never
used to train B or A+B; the Level-2 feature representations contain only the
1,440 original sample_ids. The read-only AUG-1 cache
(outputs_augmentation/features/AUG-1.npz) is used only to train A's
fold-specific SVMs when generating actor-grouped OOF train probabilities for
downstream fusion.

Protocol:
    - M4 rows are aligned to the M3 order by sample_id (never by row number);
    - B and A+B share the same MLP pipeline, search space (MLP_GRID), seed and
      training rows; StandardScaler is fitted on the train split only;
    - MLP hyperparameters are chosen on validation Macro-F1 (ties: higher
      validation accuracy, fewer parameters, larger alpha, declaration order);
    - A / B / A+B is chosen on validation Macro-F1 only (exact ties: A > B > A+B);
    - only after that selection is locked are the test rows used: B and A+B
      are evaluated once; A's test result is read from outputs_augmentation/.

Modes:
    --validate-only   check M3 (and M4 if present); no training, no output folder.
                      Without the M4 file it reports WAITING_FOR_M4.
    (default)         full comparison, written to --out-dir (outputs_visual_comparison/).

Usage:
    uv run python -m m3.run_visual_representation_comparison --validate-only
    uv run python -m m3.run_visual_representation_comparison --deep-features external/m4_visual_deep_outputs.npz --handcrafted outputs_augmentation/m3_visual_handcrafted_augmented.npz --augmentation-dir outputs_augmentation --out-dir outputs_visual_comparison --seed 0
"""
import argparse
import csv
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from m3 import run_handcrafted_ablation as abl
from m3.common import EMOTIONS
from m3.train_svm import scores

DEFAULT_HANDCRAFTED = "outputs_augmentation/m3_visual_handcrafted_augmented.npz"
DEFAULT_AUG_DIR = "outputs_augmentation"
DEFAULT_DEEP = "external/m4_visual_deep_outputs.npz"
DEFAULT_OUT = "outputs_visual_comparison"
PROTECTED_DIRS = ("outputs", "outputs_v1_global20", "outputs_ablation", "outputs_augmentation", "data",
                  "outputs_handcrafted_diagnostics")
EXPECTED_COUNTS = {"train": 960, "val": 240, "test": 240}
EXPECTED_ACTORS = {"train": 16, "val": 4, "test": 4}
# RAVDESS: odd actor ID = male, even = female; counted over unique actors per split
EXPECTED_GENDERS = {"train": {"male": 8, "female": 8}, "val": {"male": 2, "female": 2},
                    "test": {"male": 2, "female": 2}}
OOF_N_SPLITS = 4
AUG1_PURE_FLIP = {"param_flip": True, "param_rotation_deg": 0.0, "param_translate_x": 0.0,
                  "param_translate_y": 0.0, "param_scale": 1.0, "param_brightness": 1.0,
                  "param_contrast": 1.0}
AUG1_REQUIRED = ("source_sample_id", "augmented_sample_id", "label", "split", "F_pooled",
                 "config_name", "seed", "feature_dim", "inner_crop", "grid", *AUG1_PURE_FLIP)
N_FRAMES = 16
AUG_MARKERS = ("__aug", "aug01", "augmented")
REPRESENTATIONS = ("A", "B", "A+B")  # also the exact-tie order
FILE_TAG = {"A": "handcrafted", "B": "deep", "A+B": "combined"}
CLASSIFIER = {"A": "RBF SVM", "B": "MLPClassifier", "A+B": "MLPClassifier"}

HANDCRAFTED_REQUIRED = ("sample_id", "split", "label", "class_names", "F_visual_handcrafted",
                        "P_visual_handcrafted")
TIMING_KEYS = ("frame_idx", "frame_time_sec", "video_fps")

# M4 interface: role -> standard field name (overridable with --<role>-key)
M4_KEYS = {
    "sample_id": "sample_id",
    "split": "split",
    "label": "label",
    "class_names": "class_names",
    "deep_feature": "F_visual_deep",
    "deep_sequence": "F_visual_sequence",
    "deep_probability": "P_visual_deep",
    "source_model": "source_model",
}
M4_REQUIRED = ("sample_id", "split", "label", "class_names", "deep_feature")
M4_OPTIONAL_CHECKPOINT = "source_checkpoint"
ALIAS_HINTS = ("F_sequence", "F_sequence_deep", "pooled", "features", "feature", "F_deep",
               "P_deep", "probs", "probabilities", "model", "model_name", "ids", "labels")

MLP_FIXED = {"activation": "relu", "solver": "adam", "learning_rate_init": 1e-3, "batch_size": 32,
             "max_iter": 500, "early_stopping": True, "validation_fraction": 0.15,
             "n_iter_no_change": 20}
MLP_GRID = [{"hidden_layer_sizes": h, "alpha": a}
            for h in [(64,), (128,), (128, 64)] for a in [1e-4, 1e-3, 1e-2]]
MLP_TIE_RULE = ("higher validation Macro-F1; exact ties -> higher validation accuracy, then fewer "
                "MLP parameters, then larger alpha, then MLP_GRID declaration order")
REPRESENTATION_TIE_RULE = "A > B > A+B on exact ties"
CONFIGURATION_NOTE = (
    "A, B and A+B are the three complete visual configurations of the proposal (Level-2). "
    "This is not a pure feature ablation with one fixed classifier: A uses an RBF SVM "
    "(existing AUG-1 model, trained on 960 original + 960 horizontally flipped train clips); "
    "B and A+B use an MLP trained on the 960 original train clips only. Augmentation only "
    "trained each side's own model; the Level-2 concatenation uses original sample_ids only. "
    "Differences may come from the classifier and training scheme as well as the features.")

CSV_COLUMNS = [
    "representation", "classifier", "source", "original_train_samples", "augmented_train_samples",
    "total_train_samples", "feature_dim", "hidden_layer_sizes", "alpha", "svm_C", "svm_gamma",
    "svm_class_weight", "validation_macro_f1", "validation_accuracy", "random_seed",
    "fit_seconds", "n_iter", "n_parameters", "selected_config", "selected_representation",
]


class DataError(ValueError):
    """Input check failed; the message says what and where."""


# ---------------------------------------------------------------- checks

def _examples(items, k=5):
    items = list(items)
    return ", ".join(map(str, items[:k])) + (" ..." if len(items) > k else "")


def check_ids(ids, who):
    ids = list(ids)
    if len(set(ids)) != len(ids):
        seen, dup = set(), []
        for s in ids:
            if s in seen:
                dup.append(s)
            seen.add(s)
        raise DataError(f"{who}: {len(dup)} duplicate sample_id, e.g. {_examples(dup)}")
    bad = [s for s in ids if any(m in s.lower() for m in AUG_MARKERS)]
    if bad:
        raise DataError(f"{who}: augmented sample_id not allowed ({len(bad)}), e.g. {_examples(bad)}")


def check_counts(split, expected, who):
    found = {str(s): int(c) for s, c in zip(*np.unique(split, return_counts=True))}
    if found != expected:
        raise DataError(f"{who}: split counts {found}, expected {expected}")


def check_labels(label, who):
    if label.ndim != 1 or not np.issubdtype(label.dtype, np.integer):
        raise DataError(f"{who}: label must be a 1-D integer array, got {label.dtype} {label.shape}")
    if label.min() < 0 or label.max() >= len(EMOTIONS):
        raise DataError(f"{who}: label outside 0..{len(EMOTIONS) - 1} "
                        f"(min {label.min()}, max {label.max()})")


def check_class_names(names, who):
    names = [str(c) for c in np.atleast_1d(names)]
    if names != EMOTIONS:
        raise DataError(f"{who}: class_names {names} differ from the expected order {EMOTIONS}")


def check_matrix(X, n, who):
    if X.ndim != 2 or X.shape[0] != n:
        raise DataError(f"{who}: expected a 2-D array with {n} rows, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise DataError(f"{who}: contains {int((~np.isfinite(X)).sum())} NaN/Inf values")
    return X


def check_probabilities(P, n, who, atol=1e-4):
    if P.shape != (n, len(EMOTIONS)):
        raise DataError(f"{who}: expected shape ({n}, {len(EMOTIONS)}), got {P.shape}")
    if not np.isfinite(P).all():
        raise DataError(f"{who}: contains NaN/Inf")
    if P.min() < -1e-6 or P.max() > 1 + 1e-6:
        raise DataError(f"{who}: probabilities outside [0, 1] (min {P.min()}, max {P.max()})")
    dev = np.abs(P.sum(axis=1) - 1)
    if dev.max() > atol:
        raise DataError(f"{who}: rows must sum to 1 (row {int(dev.argmax())} sums to "
                        f"{P.sum(axis=1)[dev.argmax()]:.6f})")
    return P


def check_sequence(S, n, who):
    if S.ndim != 3 or S.shape[0] != n or S.shape[1] != N_FRAMES:
        raise DataError(f"{who}: expected shape ({n}, {N_FRAMES}, D_frame), got {S.shape}")
    if not np.isfinite(S).all():
        raise DataError(f"{who}: contains NaN/Inf")
    return S


def _scalar_str(v):
    v = np.asarray(v)
    if v.size != 1:
        raise DataError(f"expected a scalar string, got shape {v.shape}")
    return str(v.reshape(-1)[0])


# ------------------------------------------------------------- M3 inputs

def load_handcrafted(path, expected_counts=EXPECTED_COUNTS):
    """Validated M3 candidate export (field names as written by run_handcrafted_augmentation)."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"M3 handcrafted file not found: {path}")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in HANDCRAFTED_REQUIRED if k not in z.files]
    if missing:
        raise DataError(f"M3 {path}: missing fields {missing}; actual fields {sorted(z.files)}")
    ids = z["sample_id"].astype(str)
    n = len(ids)
    check_ids(ids, "M3")
    split, label = z["split"].astype(str), z["label"]
    if split.shape != (n,) or label.shape != (n,):
        raise DataError(f"M3: split {split.shape} / label {label.shape} do not match {n} sample_ids")
    check_counts(split, expected_counts, "M3")
    check_labels(label, "M3")
    check_class_names(z["class_names"], "M3")
    d = {"path": str(path), "fields": {k: (str(z[k].dtype), z[k].shape) for k in z.files},
         "sample_id": ids, "split": split, "label": label.astype(np.int64),
         "class_names": np.array(EMOTIONS),
         "F": check_matrix(z["F_visual_handcrafted"], n, "M3 F_visual_handcrafted"),
         "P": check_probabilities(z["P_visual_handcrafted"], n, "M3 P_visual_handcrafted")}
    if "F_sequence" in z.files:
        d["F_sequence"] = check_sequence(z["F_sequence"], n, "M3 F_sequence")
    for k in TIMING_KEYS:
        if k in z.files:
            d[k] = z[k]
    for k in ("training_augmentation", "augmentation_seed"):
        if k in z.files:
            d[k] = z[k].item()
    return d


def load_handcrafted_results(aug_dir, hand):
    """Locked A results of the AUG-1 DWT-SVM (no retraining)."""
    aug_dir = Path(aug_dir)
    paths = {k: aug_dir / k for k in ("augmentation_validation_results.csv",
                                      "selected_augmentation.json", "test_metrics.json",
                                      "confusion_matrix.csv", "selected_svm.joblib")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise DataError(f"A (AUG-1) results missing: {missing}")
    with open(paths["augmentation_validation_results.csv"]) as fh:
        rows = {r["config_name"]: r for r in csv.DictReader(fh)}
    if "AUG-1" not in rows:
        raise DataError(f"{paths['augmentation_validation_results.csv']}: no AUG-1 row")
    row = rows["AUG-1"]
    sel = json.loads(paths["selected_augmentation.json"].read_text())
    if sel["selected_config"]["config_name"] != "AUG-1":
        raise DataError(f"selected augmentation is {sel['selected_config']['config_name']}, expected AUG-1")
    tm = json.loads(paths["test_metrics.json"].read_text())
    if tm.get("config_name") != "AUG-1":
        raise DataError(f"test_metrics.json is for {tm.get('config_name')}, expected AUG-1")
    with open(paths["confusion_matrix.csv"]) as fh:
        cm_rows = list(csv.reader(fh))
    if cm_rows[0][1:] != EMOTIONS:
        raise DataError("A confusion_matrix.csv columns differ from class_names")
    cm = [[int(v) for v in r[1:]] for r in cm_rows[1:]]
    if cm != tm["test"]["confusion_matrix"]:
        raise DataError("A confusion_matrix.csv does not match test_metrics.json")
    if hand.get("training_augmentation", "horizontal_flip") != "horizontal_flip":
        raise DataError(f"M3 export training_augmentation={hand['training_augmentation']}, expected horizontal_flip")
    model = joblib.load(paths["selected_svm.joblib"])  # predict only, never refitted
    if list(model.classes_) != list(range(len(EMOTIONS))):
        raise DataError(f"A model classes_ {list(model.classes_)} != 0..7")
    if not np.allclose(model.predict_proba(hand["F"]), hand["P"], atol=1e-5):
        raise DataError("P_visual_handcrafted does not come from outputs_augmentation/selected_svm.joblib")
    original = int(row["original_train_samples"])
    augmented = int(row["augmented_train_samples"])
    return {
        "validation_macro_f1": float(row["validation_macro_f1"]),
        "validation_accuracy": float(row["validation_accuracy"]),
        "svm_C": row["best_C"], "svm_gamma": row["best_gamma"],
        "svm_class_weight": row["best_class_weight"] or None,
        "original_train_samples": original, "augmented_train_samples": augmented,
        "total_train_samples": int(row["total_train_samples"]),
        "test": tm["test"], "test_n": tm.get("n_test"),
        "model_path": str(paths["selected_svm.joblib"]),
        "model": model,  # locked final A; only cloned (same parameters) for OOF folds
        "source": str(aug_dir),
        "seed": hand.get("augmentation_seed", sel.get("protocol", {}).get("seed")),
    }


# ------------------------------------------------------------- M4 inputs

def resolve_m4_keys(fields, overrides):
    """role -> field name; explicit overrides must exist; no guessing between similar names."""
    keys = {}
    for role, std in M4_KEYS.items():
        given = overrides.get(role)
        if given is not None and given not in fields:
            raise DataError(f"--{role.replace('_', '-')}-key {given!r} not in the M4 file; "
                            f"actual fields: {sorted(fields)}")
        keys[role] = given or std
    missing = [keys[r] for r in M4_REQUIRED if keys[r] not in fields]
    if missing:
        hints = [f for f in fields if f in ALIAS_HINTS]
        msg = f"M4: missing required fields {missing}; actual fields: {sorted(fields)}."
        if hints:
            msg += (f" Possible alternatives {hints} were NOT used automatically; map them "
                    f"explicitly, e.g. --deep-feature-key <name>, --deep-sequence-key <name>.")
        raise DataError(msg)
    for role in ("deep_sequence", "deep_probability"):
        if keys[role] not in fields:
            keys[role] = None
    if keys["source_model"] not in fields:
        keys["source_model"] = None
    return keys


def resolve_source_model(npz_value, cli_value):
    """NPZ value, else --deep-source-model; both given must match exactly (after strip).

    Returns None when both are missing. A conflict raises: the command line never
    silently overrides the NPZ.
    """
    npz_value = (npz_value or "").strip() or None
    cli_value = (cli_value or "").strip() or None
    if npz_value and cli_value and npz_value != cli_value:
        raise DataError(f"source_model conflict: M4 NPZ has {npz_value!r} but --deep-source-model "
                        f"is {cli_value!r}; fix one of them (the NPZ is not overridden)")
    return npz_value or cli_value


def load_deep(path, hand, overrides=None, cli_source_model=None):
    """Validated M4 file, reordered to the M3 sample_id order."""
    path = Path(path)
    z = np.load(path, allow_pickle=False)
    keys = resolve_m4_keys(z.files, overrides or {})
    ids = z[keys["sample_id"]].astype(str)
    check_ids(ids, "M4")
    ref = list(hand["sample_id"])
    ref_set, m4_set = set(ref), set(ids)
    if ref_set != m4_set:
        missing = [s for s in ref if s not in m4_set]
        extra = [s for s in ids if s not in ref_set]
        raise DataError(f"sample_id sets differ: M3 has {len(ref)}, M4 has {len(ids)}; "
                        f"missing in M4 ({len(missing)}): {_examples(missing)}; "
                        f"extra in M4 ({len(extra)}): {_examples(extra)}")
    n = len(ids)
    F = z[keys["deep_feature"]]
    check_matrix(F, n, f"M4 {keys['deep_feature']}")
    split, label = z[keys["split"]].astype(str), z[keys["label"]]
    if split.shape != (n,) or label.shape != (n,):
        raise DataError(f"M4: split {split.shape} / label {label.shape} do not match {n} sample_ids")
    check_labels(label, "M4")
    check_class_names(z[keys["class_names"]], "M4")

    pos = {s: i for i, s in enumerate(ids)}
    order = np.array([pos[s] for s in ref])
    reordered = not np.array_equal(order, np.arange(n))
    split, label, F = split[order], label[order].astype(np.int64), F[order]
    bad = np.flatnonzero(split != hand["split"])
    if bad.size:
        i = bad[0]
        raise DataError(f"split mismatch for {bad.size} samples; first: sample_id {ref[i]} "
                        f"M3={hand['split'][i]} M4={split[i]}")
    bad = np.flatnonzero(label != hand["label"])
    if bad.size:
        i = bad[0]
        raise DataError(f"label mismatch for {bad.size} samples; first: sample_id {ref[i]} "
                        f"M3={hand['label'][i]} M4={label[i]}")
    d = {"path": str(path), "keys": keys, "fields": {k: (str(z[k].dtype), z[k].shape) for k in z.files},
         "F": F.astype(np.float32), "reordered": reordered, "P": None, "sequence": None}
    if keys["deep_probability"]:
        d["P"] = check_probabilities(z[keys["deep_probability"]], n,
                                     f"M4 {keys['deep_probability']}")[order]
    if keys["deep_sequence"]:
        d["sequence"] = check_sequence(z[keys["deep_sequence"]], n,
                                       f"M4 {keys['deep_sequence']}")[order]
    source = _scalar_str(z[keys["source_model"]]) if keys["source_model"] else None
    d["source_model"] = resolve_source_model(source, cli_source_model)
    d["source_checkpoint"] = (_scalar_str(z[M4_OPTIONAL_CHECKPOINT])
                              if M4_OPTIONAL_CHECKPOINT in z.files else None)
    return d


def build_combined(F_hand, F_deep):
    C = np.concatenate([F_hand, F_deep], axis=1)
    if C.shape[1] != F_hand.shape[1] + F_deep.shape[1]:
        raise DataError("combined dimension != handcrafted + deep")
    return C


# -------------------------------------------- actors, folds, AUG-1 cache

def extract_actor_ids(sample_ids):
    """Actor = last '-' field of the sample_id (two digits, 1..24)."""
    actors = []
    for s in sample_ids:
        last = str(s).split("-")[-1]
        if "-" not in str(s) or len(last) != 2 or not last.isdigit() or not 1 <= int(last) <= 24:
            raise DataError(f"cannot read a 1..24 actor from sample_id {s!r} (last field {last!r})")
        actors.append(int(last))
    return np.array(actors, dtype=np.int64)


def check_actor_splits(actor_id, split, expected=EXPECTED_ACTORS):
    """Actor sets of train / val / test must be disjoint (and have the expected sizes)."""
    sets = {s: set(actor_id[split == s].tolist()) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = sets[a] & sets[b]
        if both:
            raise DataError(f"actors {sorted(both)} appear in both {a} and {b}")
    if expected is not None:
        found = {s: len(v) for s, v in sets.items()}
        if found != expected:
            raise DataError(f"actors per split {found}, expected {expected}")
    return sets


def check_actor_gender_balance(actor_id, split, expected=EXPECTED_GENDERS):
    """Read-only check: male (odd) / female (even) counts over UNIQUE actors per split.

    Run after check_actor_splits (actor range, counts and disjoint sets).
    """
    found, lists = {}, {}
    for s in ("train", "val", "test"):
        actors = sorted(set(actor_id[split == s].tolist()))
        lists[s] = {"male": [a for a in actors if a % 2 == 1], "female": [a for a in actors if a % 2 == 0]}
        found[s] = {g: len(v) for g, v in lists[s].items()}
    if expected is not None and found != expected:
        lines = [f"  {s}: actual male/female={found[s]['male']}/{found[s]['female']}, "
                 f"expected {expected[s]['male']}/{expected[s]['female']}; "
                 f"male actors {lists[s]['male']}, female actors {lists[s]['female']}"
                 + ("" if found[s] == expected[s] else "  <-- differs")
                 for s in found]
        raise DataError("actor gender balance differs (odd ID = male, even = female):\n"
                        + "\n".join(lines))
    return found


def make_actor_grouped_folds(sample_ids, split, n_splits=OOF_N_SPLITS):
    """GroupKFold over train actors. Returns oof_fold [N] (-1 outside train) and per-fold actors."""
    actor_id = extract_actor_ids(sample_ids)
    tr = np.flatnonzero(split == "train")
    oof_fold = np.full(len(sample_ids), -1, dtype=np.int64)
    folds = []
    gkf = GroupKFold(n_splits=n_splits)
    for k, (fit_i, hold_i) in enumerate(gkf.split(tr, groups=actor_id[tr])):
        fit_rows, hold_rows = tr[fit_i], tr[hold_i]
        if (oof_fold[hold_rows] != -1).any():
            raise DataError(f"fold {k}: train rows assigned to two holdout folds")
        oof_fold[hold_rows] = k
        fit_a, hold_a = set(actor_id[fit_rows].tolist()), set(actor_id[hold_rows].tolist())
        if fit_a & hold_a:
            raise DataError(f"fold {k}: actors {sorted(fit_a & hold_a)} in both fit and holdout")
        folds.append({"fold": k, "fit_actors": sorted(fit_a), "holdout_actors": sorted(hold_a)})
    if (oof_fold[tr] < 0).any():
        raise DataError("some train rows received no OOF fold")
    return oof_fold, folds


def load_aug1_features(path, hand, expected_seed):
    """AUG-1 train cache aligned to the original train rows BY source_sample_id."""
    path = Path(path)
    if not path.exists():
        raise DataError(f"AUG-1 feature cache not found: {path}")
    z = np.load(path, allow_pickle=False)
    missing = [k for k in AUG1_REQUIRED if k not in z.files]
    if missing:
        raise DataError(f"AUG-1 cache {path}: missing fields {missing}; actual {sorted(z.files)}")
    src, aug_ids = z["source_sample_id"].astype(str), z["augmented_sample_id"].astype(str)
    tr = hand["split"] == "train"
    train_ids = list(hand["sample_id"][tr])
    if str(z["config_name"]) != "AUG-1":
        raise DataError(f"AUG-1 cache config_name is {z['config_name']}")
    if int(z["seed"]) != int(expected_seed):
        raise DataError(f"AUG-1 cache seed {int(z['seed'])} != selected AUG-1 seed {expected_seed}")
    for name, ids in (("source_sample_id", src), ("augmented_sample_id", aug_ids)):
        if len(set(ids)) != len(ids):
            raise DataError(f"AUG-1 cache: duplicate {name}")
    if len(src) != len(train_ids) or set(src) != set(train_ids):
        missing = [s for s in train_ids if s not in set(src)]
        extra = [s for s in src if s not in set(train_ids)]
        raise DataError(f"AUG-1 cache has {len(src)} rows, train has {len(train_ids)}; "
                        f"missing ({len(missing)}): {_examples(missing)}; extra ({len(extra)}): {_examples(extra)}")
    if set(z["split"].astype(str)) != {"train"}:
        raise DataError("AUG-1 cache contains non-train rows")
    F = z["F_pooled"]
    dim = hand["F"].shape[1]
    if F.shape != (len(train_ids), dim) or int(z["feature_dim"]) != dim:
        raise DataError(f"AUG-1 F_pooled shape {F.shape}, expected ({len(train_ids)}, {dim})")
    if not np.isfinite(F).all():
        raise DataError("AUG-1 F_pooled contains NaN/Inf")
    for k, v in (("inner_crop", 180), ("grid", 4)):
        if int(z[k]) != v:
            raise DataError(f"AUG-1 cache {k}={int(z[k])}, expected {v}")
    for k, expected in AUG1_PURE_FLIP.items():  # every row must be a pure horizontal flip
        v = np.asarray(z[k])
        if v.shape != src.shape:
            raise DataError(f"AUG-1 cache {k}: shape {v.shape}, expected {src.shape}")
        if k != "param_flip" and not np.isfinite(v.astype(np.float64)).all():
            i = int(np.flatnonzero(~np.isfinite(v.astype(np.float64)))[0])
            raise DataError(f"AUG-1 cache {k}: non-finite value {v[i]} at source_sample_id {src[i]}")
        bad = np.flatnonzero(v != expected)
        if bad.size:
            i = int(bad[0])
            raise DataError(f"AUG-1 cache is not a pure horizontal flip: {k} expected {expected}, "
                            f"first bad source_sample_id {src[i]} has {v[i]} ({bad.size} rows)")
    pos = {s: i for i, s in enumerate(src)}
    order = np.array([pos[s] for s in train_ids])  # align by ID, not by row number
    label = z["label"][order]
    y_tr = hand["label"][tr]
    bad = np.flatnonzero(label != y_tr)
    if bad.size:
        raise DataError(f"AUG-1 label mismatch for {bad.size} rows; first source_sample_id "
                        f"{train_ids[bad[0]]}: original {y_tr[bad[0]]}, AUG-1 {label[bad[0]]}")
    return {"path": str(path), "F": F[order].astype(np.float32), "source_sample_id": np.array(train_ids),
            "augmented_sample_id": aug_ids[order], "actor_id": extract_actor_ids(train_ids)}


# ------------------------------------------------------------------ OOF

def _check_fold_fit(y_fit, k, who):
    present = sorted(set(np.asarray(y_fit).tolist()))
    if present != list(range(len(EMOTIONS))):
        raise DataError(f"{who} fold {k}: fit rows lack classes "
                        f"{sorted(set(range(len(EMOTIONS))) - set(present))}")


def _canonical_proba(model, X, who):
    if list(model.classes_) != list(range(len(EMOTIONS))):
        raise DataError(f"{who}: classes_ {list(model.classes_)} != 0..7")
    return model.predict_proba(X)  # columns = classes_ = canonical order


def generate_oof_probabilities(make_model_fn, X, y, oof_fold, who):
    """Fresh model per fold, fitted on that fold's fit actors only (original train rows).

    Returns (P_oof [n_train, 8] in train-row order, per-fold info, fitted fold models).
    """
    tr = np.flatnonzero(oof_fold >= 0)
    P = np.full((len(y), len(EMOTIONS)), np.nan)
    info, models = [], []
    for k in range(int(oof_fold.max()) + 1):
        fit = tr[oof_fold[tr] != k]
        hold = np.flatnonzero(oof_fold == k)
        _check_fold_fit(y[fit], k, who)
        m = make_model_fn().fit(X[fit], y[fit])
        P[hold] = _canonical_proba(m, X[hold], f"{who} fold {k}")
        info.append({"fold": k, "fit_original_samples": int(fit.size), "fit_augmented_samples": 0,
                     "holdout_original_samples": int(hold.size)})
        models.append(m)
    P_tr = P[tr]
    if np.isnan(P_tr).any():
        raise DataError(f"{who}: some train rows have no OOF probability")
    check_probabilities(P_tr, len(tr), f"{who} OOF")
    return P_tr.astype(np.float32), info, models


def generate_augmented_svm_oof_probabilities(final_model, X_hand, y, oof_fold, sample_ids, aug):
    """A: per fold, clone the locked AUG-1 SVM (same parameters, no search) and fit it on the fit
    actors' original features + ONLY the fit actors' AUG-1 features; predict holdout originals."""
    tr = np.flatnonzero(oof_fold >= 0)
    actor = extract_actor_ids(sample_ids)
    aug_pos = {s: i for i, s in enumerate(aug["source_sample_id"])}
    P = np.full((len(y), len(EMOTIONS)), np.nan)
    info, models = [], []
    for k in range(int(oof_fold.max()) + 1):
        fit = tr[oof_fold[tr] != k]
        hold = np.flatnonzero(oof_fold == k)
        fit_actors = set(actor[fit].tolist())
        aug_rows = np.array([aug_pos[s] for s in sample_ids[fit]])  # mapped by source_sample_id
        if not set(aug["actor_id"][aug_rows].tolist()) <= fit_actors:
            raise DataError(f"A fold {k}: AUG-1 rows from holdout actors")
        X_fit = np.vstack([X_hand[fit], aug["F"][aug_rows]])
        y_fit = np.concatenate([y[fit], y[fit]])
        _check_fold_fit(y_fit, k, "A")
        m = clone(final_model).fit(X_fit, y_fit)
        P[hold] = _canonical_proba(m, X_hand[hold], f"A fold {k}")
        info.append({"fold": k, "fit_original_samples": int(fit.size),
                     "fit_augmented_samples": int(aug_rows.size),
                     "holdout_original_samples": int(hold.size),
                     "fit_augmented_ids": aug["augmented_sample_id"][aug_rows].tolist()})
        models.append(m)
    P_tr = P[tr]
    if np.isnan(P_tr).any():
        raise DataError("A: some train rows have no OOF probability")
    check_probabilities(P_tr, len(tr), "A OOF")
    return P_tr.astype(np.float32), info, models


def assemble_fusion_safe_probabilities(P_final, P_oof_train, split):
    """train rows <- OOF; val / test rows <- final locked model (a new array)."""
    tr = split == "train"
    if P_oof_train.shape != (int(tr.sum()), P_final.shape[1]):
        raise DataError(f"OOF shape {P_oof_train.shape} does not match {int(tr.sum())} train rows")
    P = P_final.astype(np.float32).copy()
    P[tr] = P_oof_train
    return check_probabilities(P, len(split), "fusion-safe probabilities")


# ------------------------------------------------------------------ MLP

def make_mlp(hidden_layer_sizes, alpha, seed):
    # StandardScaler is fitted inside the pipeline, i.e. on the training rows only.
    # early_stopping holds out validation_fraction of the TRAINING rows, not the formal val split.
    return make_pipeline(StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                                       random_state=seed, **MLP_FIXED))


def mlp_param_count(n_in, hidden, n_out=len(EMOTIONS)):
    sizes = [n_in, *hidden, n_out]
    return sum((a + 1) * b for a, b in zip(sizes, sizes[1:]))


def search_mlp(X_tr, y_tr, X_va, y_va, seed):
    """Validation-only search over MLP_GRID (no test rows are passed in).

    Returns (trials, index of the best trial, fitted best model).
    """
    trials, models = [], []
    for i, cfg in enumerate(MLP_GRID):
        t = time.perf_counter()
        m = make_mlp(cfg["hidden_layer_sizes"], cfg["alpha"], seed).fit(X_tr, y_tr)
        fit = time.perf_counter() - t
        pred = m.predict(X_va)
        trials.append({"order": i, "hidden_layer_sizes": cfg["hidden_layer_sizes"],
                       "alpha": cfg["alpha"],
                       "validation_macro_f1": f1_score(y_va, pred, average="macro"),
                       "validation_accuracy": accuracy_score(y_va, pred),
                       "fit_seconds": fit, "n_iter": int(m[-1].n_iter_),
                       "n_parameters": mlp_param_count(X_tr.shape[1], cfg["hidden_layer_sizes"])})
        models.append(m)
    best = min(range(len(trials)), key=lambda i: (
        -trials[i]["validation_macro_f1"], -trials[i]["validation_accuracy"],
        trials[i]["n_parameters"], -trials[i]["alpha"], trials[i]["order"]))
    if list(models[best].classes_) != list(range(len(EMOTIONS))):
        raise DataError(f"MLP classes_ {list(models[best].classes_)} != 0..7 "
                        "(every class must occur in the train split)")
    return trials, best, models[best]


# ------------------------------------------------------------ selection

@dataclass(frozen=True)
class SelectionLock:
    """Everything fixed on validation before any test row is used."""
    selected_representation: str
    validation_macro_f1: tuple  # ((name, f1), ...)
    deep_config: tuple
    combined_config: tuple


def select_representation(val_f1):
    """Highest validation Macro-F1; exact ties keep the earlier of A, B, A+B."""
    best = None
    for r in REPRESENTATIONS:
        if best is None or val_f1[r] > val_f1[best]:
            best = r
    return best


def lock_selection(val_f1, deep_trial, combined_trial):
    cfg = lambda t: (tuple(t["hidden_layer_sizes"]), t["alpha"])
    return SelectionLock(select_representation(val_f1),
                         tuple((r, val_f1[r]) for r in REPRESENTATIONS),
                         cfg(deep_trial), cfg(combined_trial))


def metric_summary(y, pred):
    """Overall metrics; Macro-F1 is the selection metric, Weighted-F1 is reported only."""
    return {"n_samples": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def full_scores(y, pred):
    return {**scores(y, pred), **metric_summary(y, pred)}


def evaluate_test_after_lock(lock, models, X_te, y_te, a_results):
    """Test scores once per representation; requires the validation lock.

    A keeps its archived decision rule (SVM predict(), as in outputs_augmentation); its test
    predictions are recomputed only to write per-sample reports and must reproduce the archive.
    Each entry also carries the test `predicted_label` array (not written to JSON).
    """
    if not isinstance(lock, SelectionLock):
        raise RuntimeError("test evaluation requires a SelectionLock")
    out = {}
    if "A" in X_te:
        a_pred = a_results["model"].predict(X_te["A"])
        a = full_scores(y_te, a_pred)
        archived = a_results["test"]
        if (not np.isclose(a["macro_f1"], archived["macro_f1"])
                or not np.isclose(a["accuracy"], archived["accuracy"])
                or a["confusion_matrix"] != archived["confusion_matrix"]):
            raise DataError("A test predictions do not reproduce outputs_augmentation/test_metrics.json")
        out["A"] = {**archived, "n_samples": a["n_samples"], "weighted_f1": a["weighted_f1"],
                    "source": "outputs_augmentation/test_metrics.json (reused; predictions recomputed "
                              "with the locked SVM for per-sample reports)",
                    "predicted_label": a_pred}
    else:  # minimal call without A features: archived scores only
        out["A"] = {**a_results["test"], "source": "outputs_augmentation/test_metrics.json (reused)"}
    for r in ("B", "A+B"):
        pred = models[r].predict(X_te[r])
        out[r] = {**full_scores(y_te, pred), "predicted_label": pred}
    return out


def run_oof_stage(lock, hand, X, a_results, aug, oof_fold, seed):
    """Stage 3 (after the lock): actor-grouped OOF train probabilities with LOCKED settings.

    For downstream fusion only; never used for any selection or reported as a Level-2 score.
    """
    if not isinstance(lock, SelectionLock):
        raise RuntimeError("OOF requires the validation SelectionLock")
    y, ids = hand["label"], hand["sample_id"]
    oof, info = {}, {}
    oof["A"], info["A"], _ = generate_augmented_svm_oof_probabilities(
        a_results["model"], hand["F"], y, oof_fold, ids, aug)
    for r, (hidden, alpha) in (("B", lock.deep_config), ("A+B", lock.combined_config)):
        oof[r], info[r], _ = generate_oof_probabilities(
            lambda h=hidden, a=alpha: make_mlp(h, a, seed), X[r], y, oof_fold, r)
    return oof, info


def run_comparison(hand, deep, a_results, seed, aug, oof_fold):
    split, y = hand["split"], hand["label"]
    tr, va = split == "train", split == "val"
    F_hand, F_deep = hand["F"], deep["F"]
    X = {"B": F_deep, "A+B": build_combined(F_hand, F_deep)}

    # Stage 1: validation only (train rows fit, val rows score).
    trials, best, models = {}, {}, {}
    for r in ("B", "A+B"):
        print(f"[{r}] MLP search on {int(tr.sum())} train rows, dim={X[r].shape[1]}")
        trials[r], best[r], models[r] = search_mlp(X[r][tr], y[tr], X[r][va], y[va], seed)
        b = trials[r][best[r]]
        print(f"[{r}] best hidden={b['hidden_layer_sizes']} alpha={b['alpha']} "
              f"val Macro-F1={b['validation_macro_f1']:.4f} acc={b['validation_accuracy']:.4f}")
    val_f1 = {"A": a_results["validation_macro_f1"],
              **{r: trials[r][best[r]]["validation_macro_f1"] for r in ("B", "A+B")}}
    lock = lock_selection(val_f1, trials["B"][best["B"]], trials["A+B"][best["A+B"]])
    print(f"Selected on validation (locked): {lock.selected_representation}")

    # Stage 2: test rows are used only now, once per locked model.
    te = split == "test"
    X_te = {"A": F_hand[te], **{r: X[r][te] for r in models}}
    test = evaluate_test_after_lock(lock, models, X_te, y[te], a_results)

    P = {"A": hand["P"]}
    for r in ("B", "A+B"):
        if list(models[r].classes_) != list(range(len(EMOTIONS))):
            raise DataError(f"[{r}] model classes_ differ from class_names order")
        P[r] = check_probabilities(models[r].predict_proba(X[r]).astype(np.float32),
                                   len(y), f"P_visual {r}")  # final model; train rows in-sample

    # Reported (not selection) metrics of the locked final models: train rows are ordinary
    # predictions of the final model (in-sample), not the OOF *_fusion probabilities.
    predictors = {"A": lambda m: a_results["model"].predict(F_hand[m]),
                  "B": lambda m: models["B"].predict(X["B"][m]),
                  "A+B": lambda m: models["A+B"].predict(X["A+B"][m])}
    metrics = {}
    for r in REPRESENTATIONS:
        metrics[r] = {"train": metric_summary(y[tr], predictors[r](tr)),
                      "validation": metric_summary(y[va], predictors[r](va)),
                      "test": {k: test[r][k] for k in ("n_samples", "accuracy", "macro_f1", "weighted_f1")}}
        if not np.isclose(metrics[r]["validation"]["macro_f1"], val_f1[r]):
            raise DataError(f"[{r}] validation Macro-F1 of the locked model does not match the selection")
    test_P = {r: P[r][te] for r in REPRESENTATIONS}  # final-model probabilities, never OOF

    # Stage 3: OOF train probabilities for downstream fusion (locked settings, no selection).
    oof, oof_info = run_oof_stage(lock, hand, X, a_results, aug, oof_fold, seed)
    P_fusion = {r: assemble_fusion_safe_probabilities(P[r], oof[r], split) for r in REPRESENTATIONS}
    return {"trials": trials, "best": best, "models": models, "lock": lock, "test": test,
            "P": P, "P_fusion": P_fusion, "oof": oof, "oof_info": oof_info,
            "X": {"A": F_hand, **X}, "metrics": metrics, "test_P": test_P, "test_mask": te}


# --------------------------------------------------------------- outputs

def validation_rows(res, a_results, seed, n_tr):
    sel = res["lock"].selected_representation
    rows = [{"representation": "A", "classifier": CLASSIFIER["A"], "source": a_results["source"],
             "original_train_samples": a_results["original_train_samples"],
             "augmented_train_samples": a_results["augmented_train_samples"],
             "total_train_samples": a_results["total_train_samples"],
             "feature_dim": res["X"]["A"].shape[1], "hidden_layer_sizes": "", "alpha": "",
             "svm_C": a_results["svm_C"], "svm_gamma": a_results["svm_gamma"],
             "svm_class_weight": a_results["svm_class_weight"],
             "validation_macro_f1": a_results["validation_macro_f1"],
             "validation_accuracy": a_results["validation_accuracy"],
             "random_seed": a_results["seed"],
             "fit_seconds": "", "n_iter": "", "n_parameters": "", "selected_config": True,
             "selected_representation": sel == "A"}]
    for r in ("B", "A+B"):
        for i, t in enumerate(res["trials"][r]):
            rows.append({"representation": r, "classifier": CLASSIFIER[r], "source": "level2_mlp",
                         "original_train_samples": n_tr, "augmented_train_samples": 0,
                         "total_train_samples": n_tr, "feature_dim": res["X"][r].shape[1],
                         "hidden_layer_sizes": str(t["hidden_layer_sizes"]), "alpha": t["alpha"],
                         "svm_C": "", "svm_gamma": "", "svm_class_weight": "",
                         "validation_macro_f1": t["validation_macro_f1"],
                         "validation_accuracy": t["validation_accuracy"], "random_seed": seed,
                         "fit_seconds": t["fit_seconds"], "n_iter": t["n_iter"],
                         "n_parameters": t["n_parameters"],
                         "selected_config": i == res["best"][r],
                         "selected_representation": sel == r and i == res["best"][r]})
    return rows


def write_cm(path, cm):
    """Rows = true class, columns = predicted class, both in class_names order."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["true_label \\ predicted_label", *EMOTIONS])
        for name, row in zip(EMOTIONS, cm):
            w.writerow([name, *row])


# ------------------------------------------------- evaluation reports

DISPLAY = {"A": "Handcrafted", "B": "Deep", "A+B": "Combined"}
COMPARISON_CLASSIFIER = {"A": "AUG-1 DWT-SVM", "B": "MLP", "A+B": "MLP"}
# A keeps the SVM decision rule of its archived results (predict(); Platt probabilities can
# disagree on a few samples). MLP predict() is the probability argmax.
PREDICTION_RULE = {"A": "svm_predict", "B": "argmax", "A+B": "argmax"}
COMPARISON_TYPE = "proposal-defined visual system configuration comparison"


def prob_column_names(class_names):
    cols = ["prob_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(c)).strip("_").lower() for c in class_names]
    if len(set(cols)) != len(cols):
        raise DataError(f"class names give duplicate probability columns: {cols}")
    return cols


def prediction_table(sample_ids, split, y, P, pred, rule, class_names=EMOTIONS):
    """One row per test sample, using final-model probabilities (never OOF)."""
    cols = prob_column_names(class_names)
    rows = []
    for i in range(len(sample_ids)):
        p = P[i]
        pl = int(pred[i])
        am = int(np.argmax(p))
        # confidence = probability of the predicted class (for A's SVM predict() this can differ
        # from the probability maximum, which is kept separately as argmax_confidence)
        row = {"sample_id": str(sample_ids[i]), "split": str(split[i]), "true_label": int(y[i]),
               "true_class": class_names[int(y[i])], "predicted_label": pl,
               "predicted_class": class_names[pl], "correct": bool(y[i] == pl),
               "confidence": float(p[pl]), **{c: float(v) for c, v in zip(cols, p)},
               "argmax_label": am, "argmax_confidence": float(p.max()),
               "argmax_matches_prediction": bool(am == pl), "prediction_rule": rule}
        rows.append(row)
    return rows


def check_prediction_table(rows, n_expected, class_names=EMOTIONS, atol=1e-4):
    cols = prob_column_names(class_names)
    if len(rows) != n_expected:
        raise DataError(f"prediction table has {len(rows)} rows, expected {n_expected}")
    ids = [r["sample_id"] for r in rows]
    check_ids(ids, "test predictions")
    for r in rows:
        if r["split"] != "test":
            raise DataError(f"{r['sample_id']}: split {r['split']} in test predictions")
        p = np.array([r[c] for c in cols], dtype=np.float64)
        if not np.isfinite(p).all() or abs(p.sum() - 1) > atol or p.min() < -1e-6:
            raise DataError(f"{r['sample_id']}: invalid probabilities")
        if class_names[r["true_label"]] != r["true_class"] or class_names[r["predicted_label"]] != r["predicted_class"]:
            raise DataError(f"{r['sample_id']}: label / class name mismatch")
        if (r["argmax_label"] != int(np.argmax(p)) or not np.isclose(r["argmax_confidence"], p.max())
                or not np.isclose(r["confidence"], p[r["predicted_label"]])):
            raise DataError(f"{r['sample_id']}: argmax / confidence mismatch")
        if r["prediction_rule"] == "argmax" and r["predicted_label"] != r["argmax_label"]:
            raise DataError(f"{r['sample_id']}: predicted_label is not the probability argmax")
        if r["correct"] != (r["true_label"] == r["predicted_label"]):
            raise DataError(f"{r['sample_id']}: wrong 'correct' flag")
    return ids


def confusion_from_rows(rows, n_classes=len(EMOTIONS)):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r in rows:
        cm[r["true_label"], r["predicted_label"]] += 1
    return cm


def classification_report_rows(y, pred, class_names=EMOTIONS):
    k = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=k, zero_division=0)
    rows = [{"row_type": "class", "class_index": i, "class_name": class_names[i], "precision": p[i],
             "recall": r[i], "f1_score": f[i], "support": int(s[i])} for i in k]
    n = int(s.sum())
    rows.append({"row_type": "accuracy", "class_index": "", "class_name": "", "precision": "",
                 "recall": "", "f1_score": accuracy_score(y, pred), "support": n})
    for avg, name in (("macro", "macro_avg"), ("weighted", "weighted_avg")):
        pa, ra, fa, _ = precision_recall_fscore_support(y, pred, labels=k, average=avg, zero_division=0)
        rows.append({"row_type": name, "class_index": "", "class_name": "", "precision": pa,
                     "recall": ra, "f1_score": fa, "support": n})
    return rows


def plot_confusion_matrices(cms, out_dir, class_names=EMOTIONS):
    """Raw-count 8x8 test matrices, one PNG per representation, shared colour range."""
    import matplotlib
    matplotlib.use("Agg", force=True)  # non-interactive: works without a display
    import matplotlib.pyplot as plt
    vmax = max(int(np.max(cm)) for cm in cms.values()) or 1
    paths = {}
    for r, cm in cms.items():
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        try:
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
            ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
            ax.set_yticks(range(len(class_names)), labels=class_names)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("True label")
            ax.set_title(f"{DISPLAY[r]} - test confusion matrix (counts)")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                            color="white" if cm[i, j] > vmax / 2 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = out_dir / f"confusion_matrix_{FILE_TAG[r]}.png"
            fig.savefig(path, dpi=200)
            paths[r] = path
        finally:
            plt.close(fig)
    return paths


def comparison_rows(res):
    lock = res["lock"]
    cfg = {"B": lock.deep_config, "A+B": lock.combined_config}
    rows = []
    for r in REPRESENTATIONS:
        m = res["metrics"][r]
        rows.append({"representation": FILE_TAG[r], "configuration": r,
                     "classifier": COMPARISON_CLASSIFIER[r], "feature_dim": int(res["X"][r].shape[1]),
                     "selected_hidden_layers": "" if r == "A" else str(tuple(cfg[r][0])),
                     "selected_alpha": "" if r == "A" else cfg[r][1],
                     "validation_accuracy": m["validation"]["accuracy"],
                     "validation_macro_f1": m["validation"]["macro_f1"],
                     "validation_weighted_f1": m["validation"]["weighted_f1"],
                     "test_accuracy": m["test"]["accuracy"], "test_macro_f1": m["test"]["macro_f1"],
                     "test_weighted_f1": m["test"]["weighted_f1"],
                     "selected_representation": r == lock.selected_representation,
                     "comparison_type": COMPARISON_TYPE})
    return rows


def build_reports(res, hand):
    """All test reports, validated in memory before anything is written."""
    te = res["test_mask"]
    ids, split, y = hand["sample_id"][te], hand["split"][te], hand["label"][te]
    tables, reports, cms = {}, {}, {}
    for r in REPRESENTATIONS:
        pred = np.asarray(res["test"][r]["predicted_label"])
        tables[r] = prediction_table(ids, split, y, res["test_P"][r], pred, PREDICTION_RULE[r])
        check_prediction_table(tables[r], int(te.sum()))
        cms[r] = confusion_from_rows(tables[r])
        if cms[r].tolist() != res["test"][r]["confusion_matrix"]:
            raise DataError(f"[{r}] confusion matrix does not match the prediction table")
        reports[r] = classification_report_rows(y, pred)
        acc, mac, wei = reports[r][8]["f1_score"], reports[r][9]["f1_score"], reports[r][10]["f1_score"]
        if not np.allclose([acc, mac, wei], [res["test"][r]["accuracy"], res["test"][r]["macro_f1"],
                                             res["test"][r]["weighted_f1"]]):
            raise DataError(f"[{r}] classification report differs from the test results")
        if [x["support"] for x in reports[r][:8]] != cms[r].sum(axis=1).tolist():
            raise DataError(f"[{r}] support differs from confusion-matrix row sums")
    ids0 = [x["sample_id"] for x in tables["A"]]
    if any([x["sample_id"] for x in tables[r]] != ids0 for r in REPRESENTATIONS):
        raise DataError("test prediction tables are not in the same sample order")
    return {"tables": tables, "reports": reports, "cms": cms, "comparison": comparison_rows(res)}


FUSION_FIELDS = {"A": "P_visual_handcrafted_fusion", "B": "P_visual_deep_mlp_fusion",
                 "A+B": "P_visual_combined_fusion"}
FINAL_FIELDS = {"A": "P_visual_handcrafted", "B": "P_visual_deep_mlp", "A+B": "P_visual_combined"}


def probability_protocol(res, a_results, folds, seed):
    lock = res["lock"]
    per_fold = []
    for f in folds:
        k = f["fold"]
        entry = {**f}
        for r in REPRESENTATIONS:
            i = res["oof_info"][r][k]
            entry[r] = {key: i[key] for key in ("fit_original_samples", "fit_augmented_samples",
                                                "holdout_original_samples")}
        per_fold.append(entry)
    return {
        "train_probability_protocol": "actor_grouped_4fold_oof",
        "validation_probability_protocol": "final_locked_model",
        "test_probability_protocol": "final_locked_model",
        "oof_n_splits": OOF_N_SPLITS, "oof_group_key": "actor_id",
        "oof_splitter": "sklearn GroupKFold (deterministic, no shuffling)",
        "oof_random_seed": seed,
        "folds": per_fold,
        "locked_configs": {
            "A": {"model": a_results["model_path"], "svm_params": {
                "C": a_results["svm_C"], "gamma": a_results["svm_gamma"],
                "class_weight": a_results["svm_class_weight"]},
                "oof_fit_data": "fit actors' original train features + fit actors' AUG-1 "
                                "(horizontal flip) features, matched by source_sample_id"},
            "B": {"hidden_layer_sizes": lock.deep_config[0], "alpha": lock.deep_config[1]},
            "A+B": {"hidden_layer_sizes": lock.combined_config[0], "alpha": lock.combined_config[1]},
        },
        "class_names": EMOTIONS,
        "final_model_probability_fields": FINAL_FIELDS | {"selected": "P_visual"},
        "fusion_safe_probability_fields": FUSION_FIELDS | {"selected": "P_visual_fusion"},
        "probability_fields_for_downstream_fusion": [*FUSION_FIELDS.values(), "P_visual_fusion"],
        "sources": {"train": "out-of-fold prediction of a fold model that never saw the actor",
                    "val": "final model trained on the full train split (locked on validation)",
                    "test": "final model trained on the full train split (locked on validation)"},
        "test_used_for_model_or_hyperparameter_selection": False,
        "oof_used_for_selection": False,
        "note": "OOF probabilities are inputs for downstream decision-level / gated fusion only; "
                "they are not Level-2 scores and were not used to select anything. Final-model "
                "probabilities (P_visual*, without _fusion) are in-sample on the train split.",
    }


def write_outputs(out_dir, res, hand, deep, a_results, seed, oof_fold, folds):
    """Validate every report in memory, write into a staging folder, then publish.

    Files are moved one by one (no directory-level rename, which is unreliable on Windows).
    On failure the staging folder, and an --out-dir created by this call, are removed.
    """
    out_dir = Path(out_dir)
    reports = build_reports(res, hand)
    stage = out_dir.parent / f".{out_dir.name}.partial"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    created = not out_dir.exists()
    try:
        _write_all(stage, res, hand, deep, a_results, seed, oof_fold, folds, reports)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(stage.iterdir()):  # --overwrite: same names are replaced
            os.replace(p, out_dir / p.name)
    except BaseException:
        if created:
            shutil.rmtree(out_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _write_all(out_dir, res, hand, deep, a_results, seed, oof_fold, folds, reports):
    n_train = int((hand["split"] == "train").sum())
    rows = validation_rows(res, a_results, seed, n_train)
    lock = res["lock"]
    sel = lock.selected_representation
    selection = {"selection_metric": "validation_macro_f1", "selection_split": "validation",
                 "test_used_for_selection": False,
                 "representation_tie_break": REPRESENTATION_TIE_RULE}
    mlp_config = {"pipeline": "StandardScaler (train rows only) -> MLPClassifier -> predict_proba",
                  "fixed": MLP_FIXED, "search_space": MLP_GRID, "random_state": seed,
                  "tie_break": MLP_TIE_RULE}
    deep_info = {"path": deep["path"], "source_model": deep["source_model"],
                 "source_checkpoint": deep["source_checkpoint"], "field_mapping": deep["keys"],
                 "deep_features_reordered": deep["reordered"],
                 "deep_sequence_available": deep["sequence"] is not None,
                 "deep_probability_available": deep["P"] is not None}

    with open(out_dir / "validation_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "validation_results.json", "w") as fh:
        json.dump({"note": CONFIGURATION_NOTE, **selection, "mlp_config": mlp_config,
                   "deep_input": deep_info, "results": rows}, fh, indent=2, default=str)
    best_cfg = {r: res["trials"][r][res["best"][r]] for r in ("B", "A+B")}
    with open(out_dir / "selected_configs.json", "w") as fh:
        json.dump({
            "selected_representation": sel, "selected_classifier": CLASSIFIER[sel],
            "selected_feature_dim": int(res["X"][sel].shape[1]),
            "validation_macro_f1": dict(lock.validation_macro_f1), **selection,
            "configurations": {
                "A": {"classifier": CLASSIFIER["A"], "model_reference": a_results["model_path"],
                      "svm_params": {"C": a_results["svm_C"], "gamma": a_results["svm_gamma"],
                                     "class_weight": a_results["svm_class_weight"]},
                      "train_samples": {"original": a_results["original_train_samples"],
                                        "augmented": a_results["augmented_train_samples"],
                                        "total": a_results["total_train_samples"]},
                      "retrained": False},
                "B": {"classifier": CLASSIFIER["B"], "best_mlp": best_cfg["B"],
                      "model_file": "best_deep_mlp.joblib", "train_samples": n_train},
                "A+B": {"classifier": CLASSIFIER["A+B"], "best_mlp": best_cfg["A+B"],
                        "model_file": "best_combined_mlp.joblib", "train_samples": n_train},
            },
            "mlp_config": mlp_config, "mlp_hyperparameter_tie_break": MLP_TIE_RULE,
            "deep_input": deep_info, "note": CONFIGURATION_NOTE,
            "probability_protocol": "probability_protocol.json (downstream fusion: *_fusion fields)",
        }, fh, indent=2, default=str)
    with open(out_dir / "test_results.json", "w") as fh:
        json.dump({"selection_basis": "validation_macro_f1", "test_used_for_selection": False,
                   "test_evaluated_after_selection_lock": True, "selected_representation": sel,
                   "n_test": int((hand["split"] == "test").sum()),
                   "results": {r: {k: v for k, v in res["test"][r].items() if k != "predicted_label"}
                               for r in REPRESENTATIONS},
                   "summary_metrics": res["metrics"],
                   "metric_notes": {
                       "selection": "validation Macro-F1 only; Weighted-F1 and test metrics are reported, never used to select",
                       "train": "ordinary predictions of the locked final model on the original train rows "
                                "(in-sample); not the OOF *_fusion probabilities",
                       "A_decision_rule": "SVM predict(), as in the archived AUG-1 results",
                       "comparison_type": COMPARISON_TYPE},
                   "note": "A test result reused from outputs_augmentation (predictions recomputed with the "
                           "locked SVM only for per-sample reports). The D/AUG-1 test results had been "
                           "observed during earlier development."},
                  fh, indent=2, default=str)
    for r in REPRESENTATIONS:
        tag = FILE_TAG[r]
        write_cm(out_dir / f"confusion_matrix_{tag}.csv", reports["cms"][r].tolist())
        _write_rows(out_dir / f"classification_report_{tag}.csv", reports["reports"][r])
        _write_rows(out_dir / f"test_predictions_{tag}.csv", reports["tables"][r])
    _write_rows(out_dir / "representation_comparison.csv", reports["comparison"])
    plot_confusion_matrices(reports["cms"], out_dir)
    for name, artifact in mlp_artifacts(res, deep, seed).items():
        joblib.dump(artifact, out_dir / name)

    extra = {}
    if "F_sequence" in hand:
        extra["F_visual_handcrafted_sequence"] = hand["F_sequence"].astype(np.float32)
    if deep["sequence"] is not None:
        extra["F_visual_deep_sequence"] = deep["sequence"]  # from M4 (F_visual_sequence)
    if deep["P"] is not None:
        extra["P_visual_deep_source"] = deep["P"].astype(np.float32)  # M4's own classifier
    if deep["source_checkpoint"]:
        extra["deep_source_checkpoint"] = deep["source_checkpoint"]
    for k in TIMING_KEYS:
        if k in hand:
            extra[k] = hand[k]
    protocol = probability_protocol(res, a_results, folds, seed)
    with open(out_dir / "probability_protocol.json", "w") as fh:
        json.dump(protocol, fh, indent=2, default=str)
    np.savez(
        out_dir / "visual_representation_outputs.npz",
        sample_id=hand["sample_id"], split=hand["split"], label=hand["label"],
        class_names=np.array(EMOTIONS),
        F_visual_handcrafted=res["X"]["A"].astype(np.float32),
        F_visual_deep=res["X"]["B"].astype(np.float32),
        F_visual_combined=res["X"]["A+B"].astype(np.float32),
        P_visual_handcrafted=res["P"]["A"], P_visual_deep_mlp=res["P"]["B"],
        P_visual_combined=res["P"]["A+B"],
        selected_representation=sel, selected_classifier=CLASSIFIER[sel],
        selected_feature_dim=int(res["X"][sel].shape[1]),
        F_visual=res["X"][sel].astype(np.float32), P_visual=res["P"][sel],
        deep_source_model=deep["source_model"],
        deep_sequence_available=deep["sequence"] is not None,
        selection_metric="validation_macro_f1", selection_split="validation",
        test_used_for_selection=False, level2_random_seed=seed,
        level2_mlp_config=json.dumps(mlp_config, default=str),
        # fusion-safe: train rows = actor-grouped OOF, val/test rows = final locked model
        **{FUSION_FIELDS[r]: res["P_fusion"][r] for r in REPRESENTATIONS},
        P_visual_fusion=res["P_fusion"][sel],
        actor_id=extract_actor_ids(hand["sample_id"]), oof_fold=oof_fold,
        train_probability_protocol="actor_grouped_4fold_oof",
        validation_probability_protocol="final_locked_model",
        test_probability_protocol="final_locked_model",
        oof_n_splits=OOF_N_SPLITS, oof_group_key="actor_id", oof_random_seed=seed,
        probability_fields_for_downstream_fusion=np.array(
            protocol["probability_fields_for_downstream_fusion"]),
        **extra,
    )
    write_readme(out_dir, res, hand, deep, a_results, rows, reports, seed)


def mlp_artifacts(res, deep, seed):
    """Self-contained MLP joblibs: the sklearn Pipeline under "model" plus what is needed to use it.

    No feature arrays are stored. Load with `joblib.load(path)["model"]`.
    """
    lock = res["lock"]
    dims = {r: int(res["X"][r].shape[1]) for r in REPRESENTATIONS}
    common = {"class_names": list(EMOTIONS), "random_seed": seed,
              "source_model": deep["source_model"], "source_checkpoint": deep["source_checkpoint"]}
    deep_art = {"model": res["models"]["B"], "representation": "B", "representation_name": "deep",
                "feature_dim": dims["B"], **common, "input_fields": ["F_visual_deep"],
                "selected_hidden_layers": tuple(lock.deep_config[0]),
                "selected_alpha": lock.deep_config[1]}
    order = ["F_visual_handcrafted", "F_visual_deep"]
    combined_art = {"model": res["models"]["A+B"], "representation": "A+B",
                    "representation_name": "combined", "feature_dim": dims["A+B"],
                    "handcrafted_feature_dim": dims["A"], "deep_feature_dim": dims["B"], **common,
                    "input_fields": list(order), "concatenation_order": list(order),
                    "concatenation_axis": 1,
                    "selected_hidden_layers": tuple(lock.combined_config[0]),
                    "selected_alpha": lock.combined_config[1]}
    if combined_art["handcrafted_feature_dim"] + combined_art["deep_feature_dim"] != combined_art["feature_dim"]:
        raise DataError("combined artifact dimensions do not add up")
    return {"best_deep_mlp.joblib": deep_art, "best_combined_mlp.joblib": combined_art}


def _write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_readme(out_dir, res, hand, deep, a_results, rows, reports, seed):
    sel = res["lock"].selected_representation
    m = res["metrics"]
    lock = res["lock"]
    dims = {r: int(res["X"][r].shape[1]) for r in REPRESENTATIONS}
    better = {base: m["A+B"]["test"]["macro_f1"] > m[r]["test"]["macro_f1"]
              for base, r in (("handcrafted", "A"), ("deep", "B"))}
    val_better = {base: m["A+B"]["validation"]["macro_f1"] > m[r]["validation"]["macro_f1"]
                  for base, r in (("handcrafted", "A"), ("deep", "B"))}
    lines = [
        "# Level-2 visual representation comparison (A / B / A+B)", "",
        f"This is a {COMPARISON_TYPE}. {CONFIGURATION_NOTE}", "",
        f"- M4 `source_model`: **{deep['source_model']}**"
        + (f" (checkpoint `{deep['source_checkpoint']}`)" if deep["source_checkpoint"] else "")
        + f"; deep features `{deep['path']}`, reordered by sample_id: {deep['reordered']}.", "",
        "## Configurations", "",
        "| Configuration | Input | Dim | Classifier | Training rows |", "|---|---|---|---|---|",
        f"| Handcrafted (A) | `F_visual_handcrafted` (Full-320 DWT) | {dims['A']} | AUG-1 DWT-SVM "
        f"(RBF SVC, C={a_results['svm_C']}, gamma={a_results['svm_gamma']}, "
        f"class_weight={a_results['svm_class_weight']}; existing model, not retrained) | "
        f"{a_results['original_train_samples']} original + {a_results['augmented_train_samples']} flipped |",
        f"| Deep (B) | `F_visual_deep` | {dims['B']} | MLP, hidden={tuple(lock.deep_config[0])}, "
        f"alpha={lock.deep_config[1]} | {rows[1]['total_train_samples']} original |",
        f"| Combined (A+B) | `[F_visual_handcrafted, F_visual_deep]` | {dims['A+B']} | MLP, "
        f"hidden={tuple(lock.combined_config[0])}, alpha={lock.combined_config[1]} | "
        f"{rows[1]['total_train_samples']} original |", "",
        f"MLP search space (same for B and A+B, seed {seed}): hidden_layer_sizes "
        f"{[c['hidden_layer_sizes'] for c in MLP_GRID][::3]} x alpha {sorted({c['alpha'] for c in MLP_GRID})}; "
        f"fixed {MLP_FIXED}. Each MLP and the A / B / A+B choice are selected on **validation Macro-F1** "
        f"(MLP ties: {MLP_TIE_RULE}; representation ties: {REPRESENTATION_TIE_RULE}). Test rows were used "
        "only after the selection was locked; Weighted-F1 and test metrics are reported, never used to select.", "",
        f"**Selected visual system (validation): {DISPLAY[sel]} ({sel})**, classifier "
        f"{COMPARISON_CLASSIFIER[sel]}, dim {dims[sel]}.", "",
        "## Main comparison", "",
        "| Configuration | Val Acc | Val Macro-F1 | Val Weighted-F1 | Test Acc | Test Macro-F1 | Test Weighted-F1 |",
        "|---|---|---|---|---|---|---|",
        *[f"| {DISPLAY[r]} | {m[r]['validation']['accuracy']:.4f} | {m[r]['validation']['macro_f1']:.4f} | "
          f"{m[r]['validation']['weighted_f1']:.4f} | {m[r]['test']['accuracy']:.4f} | "
          f"{m[r]['test']['macro_f1']:.4f} | {m[r]['test']['weighted_f1']:.4f} |" for r in REPRESENTATIONS], "",
        f"- Combined vs Handcrafted: validation Macro-F1 {'higher' if val_better['handcrafted'] else 'not higher'}, "
        f"test Macro-F1 {'higher' if better['handcrafted'] else 'not higher'}.",
        f"- Combined vs Deep: validation Macro-F1 {'higher' if val_better['deep'] else 'not higher'}, "
        f"test Macro-F1 {'higher' if better['deep'] else 'not higher'}.",
        "- Train metrics (in `test_results.json`, `summary_metrics`) are ordinary in-sample predictions "
        "of the final models, not the OOF `_fusion` probabilities.", "",
        "## Files", "",
        "- `representation_comparison.csv`: the three configurations (classifier, dim, selected MLP "
        "settings, validation / test Accuracy, Macro-F1, Weighted-F1).",
        "- `classification_report_{handcrafted,deep,combined}.csv`: per-class precision / recall / F1 / "
        "support on the 240 test clips, then accuracy, macro_avg, weighted_avg rows.",
        "- `test_predictions_{handcrafted,deep,combined}.csv`: one row per test clip (same order in all "
        "three) with true / predicted label, confidence and the 8 final-model probabilities. B and A+B "
        "predict the probability argmax; A uses the SVM `predict()` rule of its archived results "
        "(`argmax_label` / `argmax_matches_prediction` show where Platt probabilities disagree).",
        "- `confusion_matrix_{handcrafted,deep,combined}.csv` / `.png`: raw test counts, rows = true "
        "label, columns = predicted label, same colour range in the three images.",
        f"- A reuses the existing AUG-1 SVM read-only (`{a_results['model_path']}`); "
        "`best_deep_mlp.joblib`, `best_combined_mlp.joblib`: locked MLPs saved as dictionary "
        "artifacts; `artifact[\"model\"]` is the sklearn Pipeline, the other keys record the "
        "representation, input fields and dimensions, class order, random seed, M4 source model "
        "and (Combined) the concatenation order. The A SVM stays a plain model file, not a dictionary.",
        "",
        "```python",
        "import joblib",
        "import numpy as np",
        "",
        "deep_artifact = joblib.load(\"best_deep_mlp.joblib\")",
        "deep_model = deep_artifact[\"model\"]",
        "deep_probabilities = deep_model.predict_proba(F_visual_deep)",
        "",
        "combined_artifact = joblib.load(\"best_combined_mlp.joblib\")",
        "combined_model = combined_artifact[\"model\"]",
        "# order from combined_artifact[\"concatenation_order\"]: F_visual_handcrafted, then F_visual_deep",
        "F_visual_combined = np.concatenate(",
        "    [F_visual_handcrafted, F_visual_deep],",
        "    axis=combined_artifact[\"concatenation_axis\"],",
        ")",
        "combined_probabilities = combined_model.predict_proba(F_visual_combined)",
        "```",
        "- `visual_representation_outputs.npz`, `probability_protocol.json`: features and probabilities "
        "for fusion.", "",
        "## Probabilities for downstream fusion", "",
        "- Final-model probabilities (`P_visual_handcrafted`, `P_visual_deep_mlp`, `P_visual_combined`, "
        "`P_visual`) are in-sample on the train split; the test reports above use them on test only.",
        "- **Decision-level / gated fusion must use the fusion-safe fields** (`P_visual_handcrafted_fusion`, "
        "`P_visual_deep_mlp_fusion`, `P_visual_combined_fusion`, `P_visual_fusion`): train rows are "
        "actor-grouped 4-fold OOF, val / test rows come from the final locked model. OOF probabilities "
        "were not used for any selection and are not Level-2 scores.",
        "- `F_visual` / `P_visual` / `P_visual_fusion` belong to the selected configuration; fusion must "
        "align on `sample_id`.", "",
        "## Limitations", "",
        "- A uses an SVM (trained with horizontally flipped copies), B and A+B use an MLP (original clips "
        "only), so this is not a strictly controlled experiment that changes only the representation.",
        "- Test results are held-out results under the fixed actor-independent split; earlier D / AUG-1 "
        "test results had already been observed, so interpret them cautiously.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ main

def parse_counts(s):
    try:
        d = {k: int(v) for k, v in (p.split("=") for p in s.split(","))}
    except ValueError:
        raise SystemExit(f"--expected-counts must look like train=960,val=240,test=240, got {s!r}")
    if set(d) != {"train", "val", "test"}:
        raise SystemExit("--expected-counts needs train, val and test")
    return d


def parse_gender_counts(s):
    try:
        d = {}
        for part in s.split(","):
            k, v = part.split("=")
            m, f = v.split(":")
            d[k] = {"male": int(m), "female": int(f)}
    except ValueError:
        raise SystemExit(f"--expected-gender-counts must look like train=8:8,val=2:2,test=2:2, got {s!r}")
    if set(d) != {"train", "val", "test"}:
        raise SystemExit("--expected-gender-counts needs train, val and test")
    return d


def print_waiting(deep_path):
    print("STATUS: WAITING_FOR_M4")
    print(f"  Expected M4 file : {deep_path}")
    print("  Required data fields:")
    for f in ("sample_id [N]", "split [N]", "label [N]", "class_names [8]", "F_visual_deep [N, D]"):
        print(f"    - {f}")
    print("  Required provenance for formal run:")
    print("    - source_model in the NPZ, OR provide --deep-source-model")
    print("  Optional fields:")
    for f in (f"F_visual_sequence [N, {N_FRAMES}, D_frame]", "P_visual_deep [N, 8]",
              M4_OPTIONAL_CHECKPOINT):
        print(f"    - {f}")
    print("  The formal Level-2 experiment has NOT been run; no model trained, no scores produced.")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--deep-features", default=DEFAULT_DEEP)
    ap.add_argument("--handcrafted", default=DEFAULT_HANDCRAFTED)
    ap.add_argument("--augmentation-dir", default=DEFAULT_AUG_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate-only", action="store_true",
                    help="check inputs only; no training, no output folder")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace results previously written by this program in --out-dir")
    ap.add_argument("--deep-source-model", default=None,
                    help="M4 source model name if the NPZ has no source_model; must equal the "
                         "NPZ value when both are given")
    ap.add_argument("--expected-counts", default="train=960,val=240,test=240")
    ap.add_argument("--expected-actor-counts", default="train=16,val=4,test=4")
    ap.add_argument("--expected-gender-counts", default="train=8:8,val=2:2,test=2:2",
                    help="male:female unique actors per split")
    ap.add_argument("--aug1-features", default=None,
                    help="AUG-1 train feature cache (default <augmentation-dir>/features/AUG-1.npz)")
    for role in M4_KEYS:
        ap.add_argument(f"--{role.replace('_', '-')}-key", dest=f"key_{role}", default=None,
                        help=f"M4 field for {role} (default {M4_KEYS[role]})")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return _main(args)
    except DataError as e:
        raise SystemExit(f"ERROR: {e}") from None


def _main(args):
    counts = parse_counts(args.expected_counts)
    hand = load_handcrafted(args.handcrafted, counts)
    a_results = load_handcrafted_results(args.augmentation_dir, hand)
    print(f"M3 OK: {hand['path']}  N={len(hand['sample_id'])}  splits={counts}  "
          f"F_visual_handcrafted={hand['F'].shape}  P_visual_handcrafted={hand['P'].shape}")
    print(f"A (AUG-1 DWT-SVM, reused): val Macro-F1={a_results['validation_macro_f1']:.4f} "
          f"acc={a_results['validation_accuracy']:.4f}")
    # M3-side inputs of the OOF stage are checked in every mode (no model is fitted here).
    actor_id = extract_actor_ids(hand["sample_id"])
    sets = check_actor_splits(actor_id, hand["split"], parse_counts(args.expected_actor_counts))
    print("Actor splits OK: train/val/test = "
          + "/".join(str(len(sets[s])) for s in ("train", "val", "test")) + " actors, disjoint")
    g = check_actor_gender_balance(actor_id, hand["split"],
                                   parse_gender_counts(args.expected_gender_counts))
    print("Actor gender balance OK:")
    print(f"  train male/female={g['train']['male']}/{g['train']['female']}, "
          f"val={g['val']['male']}/{g['val']['female']}, test={g['test']['male']}/{g['test']['female']}")
    oof_fold, folds = make_actor_grouped_folds(hand["sample_id"], hand["split"])
    print(f"OOF folds: GroupKFold(n_splits={OOF_N_SPLITS}) over train actors; holdout actors "
          f"{[f['holdout_actors'] for f in folds]}")
    aug1_path = Path(args.aug1_features or Path(args.augmentation_dir) / "features" / "AUG-1.npz")
    aug = load_aug1_features(aug1_path, hand, a_results["seed"])
    print(f"AUG-1 cache OK: {aug1_path} ({len(aug['F'])} rows, pure horizontal flip)")

    deep_path = Path(args.deep_features)
    if not deep_path.exists():
        print_waiting(deep_path)
        if args.validate_only:
            return {"status": "WAITING_FOR_M4"}
        raise SystemExit(f"WAITING_FOR_M4: {deep_path} not found; nothing was trained or written.")

    overrides = {r: getattr(args, f"key_{r}") for r in M4_KEYS}
    deep = load_deep(deep_path, hand, overrides, args.deep_source_model)
    combined_dim = hand["F"].shape[1] + deep["F"].shape[1]
    print(f"M4 OK: {deep_path}  reordered by sample_id: {deep['reordered']}")
    print(f"  shapes: handcrafted {hand['F'].shape}, deep {deep['F'].shape}, "
          f"combined ({len(hand['sample_id'])}, {combined_dim})")
    print(f"  F_visual_sequence: {'present' if deep['sequence'] is not None else 'not provided'}; "
          f"P_visual_deep: {'present' if deep['P'] is not None else 'not provided'}")
    print(f"  source_model: {deep['source_model'] or 'MISSING'}")

    if args.validate_only:
        if not deep["source_model"]:
            print("WARNING: source_model is MISSING (not in the M4 NPZ and no --deep-source-model); "
                  "the formal Level-2 run cannot be run yet.")
        print("STATUS: M4_VALID (validate-only: nothing trained, no test used, nothing written)")
        return {"status": "M4_VALID", "deep_features_reordered": deep["reordered"],
                "combined_dim": combined_dim, "source_model": deep["source_model"]}

    if not deep["source_model"]:
        raise SystemExit("ERROR: source_model missing; add it to the M4 NPZ or pass "
                         "--deep-source-model. No results written.")
    out_dir = Path(args.out_dir)
    abl.check_out_dir(out_dir, args.augmentation_dir, args.overwrite, protected=PROTECTED_DIRS,
                      marker="selected_configs.json")
    # Everything is computed in memory first; the output folder is created only when writing.
    res = run_comparison(hand, deep, a_results, args.seed, aug, oof_fold)
    write_outputs(out_dir, res, hand, deep, a_results, args.seed, oof_fold, folds)
    print(f"Done -> {out_dir}")
    return {"status": "DONE", "selected_representation": res["lock"].selected_representation}


if __name__ == "__main__":
    main()
