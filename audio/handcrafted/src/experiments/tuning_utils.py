
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT / "outputs/features/experimental_v2/combined"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/metrics/experimental_v2/tuning"
)

RANDOM_STATE = 42

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]

EMOTION_LABELS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

LABEL_MAP = {
    label: index
    for index, label in enumerate(EMOTION_LABELS)
}


def load_dataset(
    wavelet: str,
    normalization: str,
    frame_ms: int,
):
    """Load a 48D dataset and separate train/validation."""

    file_path = (
        FEATURE_DIR
        / wavelet
        / f"combined_{normalization}_{frame_ms}ms.csv"
    )

    dataset = pd.read_csv(file_path)

    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]

    y_train = train_df["emotion_label"].map(LABEL_MAP)
    y_val = val_df["emotion_label"].map(LABEL_MAP)

    groups = train_df["actor_id"].to_numpy()

    # Validation checks
    assert len(feature_columns) == 48
    assert len(dataset) == 1440
    assert dataset["sample_id"].is_unique

    assert len(train_df) == 960
    assert len(val_df) == 240

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    assert y_train.notna().all()
    assert y_val.notna().all()

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    return (
        X_train,
        y_train.astype(int),
        X_val,
        y_val.astype(int),
        groups,
    )


def create_cv():
    """Create actor-independent cross-validation folds."""

    return StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate a fitted classifier."""

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_macro_f1 = f1_score(
        y_train,
        train_pred,
        average="macro",
        zero_division=0,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )

    return {
        "train_accuracy": accuracy_score(
            y_train, train_pred
        ),
        "train_macro_f1": train_macro_f1,
        "val_accuracy": accuracy_score(
            y_val, val_pred
        ),
        "val_macro_f1": val_macro_f1,
        "f1_gap": train_macro_f1 - val_macro_f1,
    }


def save_search_results(search, model_name: str):
    """Save model-specific hyperparameter search results."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(search.cv_results_)

    output_path = (
        OUTPUT_DIR / f"{model_name}_search_results.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("Search results saved:", output_path)

    return OUTPUT_DIR