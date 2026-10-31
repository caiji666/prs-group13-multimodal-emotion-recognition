"""Step 3: DWT-SVM baseline (Configuration A) + export for M5.

Proposal Sec. 3: feature standardisation -> SVM, 8-class.
Hyperparameters are chosen on the validation set (Macro-F1); the test set is
only used once for the final report.

Outputs (in --out-dir):
    m3_visual_handcrafted.npz   the file to send to M5 (fields in README.md)
    metrics.json                val/test scores, best params, confusion matrices
    dwt_svm.joblib              fitted StandardScaler + SVM pipeline (joblib.load)

Usage:
    uv run python -m m3.train_svm --features data/dwt_features.npz --out-dir outputs
"""
import argparse
import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from m3.common import EMOTIONS

# sklearn 1.9 deprecates SVC(probability=True) (removed in 1.11); still correct here.
warnings.filterwarnings("ignore", message=".*`probability` parameter was deprecated.*")

TIMING_KEYS = ("frame_idx", "frame_time_sec", "video_fps")

GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.001, 0.01, 0.1],
    "class_weight": [None, "balanced"],
}


def make_model(C, gamma, class_weight, random_state=0):
    # The scaler is fitted on the training data only (inside the pipeline).
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
            probability=True, random_state=random_state),
    )


def select_hyperparams(X_tr, y_tr, X_va, y_va, random_state=0, verbose=True):
    """Grid search on validation Macro-F1 only; ties keep the first combination in GRID order.

    Only train and validation arrays are passed in, so the test set cannot
    influence the choice. Returns ((C, gamma, class_weight), best val Macro-F1).
    """
    best, best_f1 = None, -1
    for C, gamma, cw in itertools.product(*GRID.values()):
        m = make_model(C, gamma, cw, random_state).fit(X_tr, y_tr)
        f1 = f1_score(y_va, m.predict(X_va), average="macro")
        if verbose:
            print(f"C={C:<5} gamma={gamma!s:<6} class_weight={cw!s:<9} val Macro-F1={f1:.4f}")
        if f1 > best_f1:
            best, best_f1 = (C, gamma, cw), f1
    return best, best_f1


def scores(y, pred):
    return {
        "macro_f1": f1_score(y, pred, average="macro"),
        "accuracy": accuracy_score(y, pred),
        "report": classification_report(y, pred, labels=range(8), target_names=EMOTIONS,
                                        output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(8)).tolist(),
    }


def check(d):
    ids = d["sample_id"]
    assert len(ids) == len(set(ids)), "duplicate sample_id"
    counts = {s: int((d["split"] == s).sum()) for s in ("train", "val", "test")}
    print("Samples per split:", counts, "(expected 960 / 240 / 240)")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/dwt_features.npz")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args(argv)

    d = dict(np.load(args.features))
    missing = [k for k in TIMING_KEYS if k not in d]
    if missing:
        raise KeyError(f"{args.features} has no {missing}; re-run m3.dwt_features")
    counts = check(d)
    X, y, split = d["F_pooled"], d["label"], d["split"]
    tr, va, te = split == "train", split == "val", split == "test"

    best, best_f1 = select_hyperparams(X[tr], y[tr], X[va], y[va])
    print(f"Best: C={best[0]} gamma={best[1]} class_weight={best[2]} (val Macro-F1={best_f1:.4f})")

    model = make_model(*best).fit(X[tr], y[tr])
    assert list(model.classes_) == list(range(8))
    # Metrics use predict(), the same rule as the val Macro-F1 used for tuning.
    # predict_proba() (Platt scaling) can disagree with predict() on some samples,
    # so it is only used for the exported probabilities.
    pred = model.predict(X)
    P = model.predict_proba(X)  # columns follow model.classes_ = 0..7 = EMOTIONS order

    metrics = {
        "best_params": {"C": best[0], "gamma": best[1], "class_weight": best[2]},
        "counts": counts,
        "val": scores(y[va], pred[va]),
        "test": scores(y[te], pred[te]),
    }
    print(f"VAL  Macro-F1={metrics['val']['macro_f1']:.4f}  Acc={metrics['val']['accuracy']:.4f}")
    print(f"TEST Macro-F1={metrics['test']['macro_f1']:.4f}  Acc={metrics['test']['accuracy']:.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n, t = d["F_sequence"].shape[:2]
    np.savez(
        out / "m3_visual_handcrafted.npz",
        sample_id=d["sample_id"],
        split=d["split"],
        label=y,
        F_sequence=d["F_sequence"].astype(np.float32),
        sequence_length=np.full(n, t, dtype=np.int64),
        padding_mask=np.zeros((n, t), dtype=bool),  # True = padding; never padded here
        F_visual_handcrafted=X.astype(np.float32),
        P_visual_handcrafted=P.astype(np.float32),
        class_names=np.array(EMOTIONS),
        feature_names=d["feature_names"],
        **{k: d[k] for k in TIMING_KEYS},  # passed through unchanged
    )
    with open(out / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    joblib.dump(model, out / "dwt_svm.joblib")  # full StandardScaler + SVM pipeline
    print(f"Saved -> {out / 'm3_visual_handcrafted.npz'}, metrics.json, dwt_svm.joblib")


if __name__ == "__main__":
    main()
