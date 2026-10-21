# Train deep-only and combined-feature MLP classifiers

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

M1_PATH = (
    PROJECT_ROOT / "outputs/features/handcrafted_features.csv"
)

M2_DIR = PROJECT_ROOT / "data/external/m2_audio/pooled"

OUTPUT_DIR = PROJECT_ROOT / "outputs/metrics"

RANDOM_SEED = 42

LABEL_MAP = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprise": 7,
}


def load_aligned_features():
    """Load and align M1 and M2 features by recording ID."""

    m1 = pd.read_csv(M1_PATH)

    m2 = pd.concat(
        [
            pd.read_csv(M2_DIR / f"{split}.csv")
            for split in ("train", "val", "test")
        ],
        ignore_index=True,
    ).copy()

    m1["m2_sample_id"] = (
        m1["sample_id"]
        .str.split("-")
        .str[1:]
        .str.join("-")
    )

    m1["label"] = m1["emotion_label"].map(LABEL_MAP)

    if m1["label"].isna().any():
        raise ValueError("Unknown emotion label")

    if not m1["m2_sample_id"].is_unique:
        raise ValueError("Duplicate M1 IDs")

    if not m2["sample_id"].is_unique:
        raise ValueError("Duplicate M2 IDs")

    merged = m1.merge(
        m2,
        left_on="m2_sample_id",
        right_on="sample_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_m1", "_m2"),
    )

    assert (merged["_merge"] == "both").all()
    assert len(merged) == 1440

    for column in ("actor_id", "split", "label"):
        assert (
            merged[f"{column}_m1"]
            == merged[f"{column}_m2"]
        ).all(), f"{column} mismatch"

    return merged, m1


def get_feature_groups(merged, m1):
    """Create deep-only and combined feature configurations."""

    metadata_cols = {
        "sample_id", "trial_id", "actor_id",
        "emotion_label", "split", "m2_sample_id",
        "label",
    }

    handcrafted_cols = [
        c for c in m1.columns
        if c not in metadata_cols
    ]

    reduced_cols = [
        c for c in handcrafted_cols
        if not (
            c.startswith("dwt_")
            and c.endswith("_energy")
        )
    ]

    deep_cols = [f"f_{i:03d}" for i in range(128)]

    groups = {
        "transformer_only": deep_cols,
        "handcrafted_transformer": handcrafted_cols + deep_cols,
        "reduced_transformer": reduced_cols + deep_cols,
    }

    expected = {
        "transformer_only": 128,
        "handcrafted_transformer": 176,
        "reduced_transformer": 171,
    }

    for name, columns in groups.items():
        assert len(columns) == expected[name]
        assert set(columns).issubset(merged.columns)

    return groups


def create_mlp():
    """Create the same MLP configuration for each experiment."""

    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=RANDOM_SEED,
        )),
    ])


def main():
    merged, m1 = load_aligned_features()
    groups = get_feature_groups(merged, m1)

    train_df = merged[merged["split_m1"] == "train"]
    val_df = merged[merged["split_m1"] == "val"]

    y_train = train_df["label_m1"].astype(int)
    y_val = val_df["label_m1"].astype(int)

    results = []

    for name, columns in groups.items():

        print(f"\nTraining: {name}")

        X_train = train_df[columns]
        X_val = val_df[columns]

        assert np.isfinite(X_train.to_numpy()).all()
        assert np.isfinite(X_val.to_numpy()).all()

        model = create_mlp()

        model.fit(X_train, y_train)

        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append({
            "experiment": name,
            "n_features": len(columns),
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
        })

        print(
            f"Accuracy: {accuracy:.4f} | "
            f"Macro-F1: {macro_f1:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "val_macro_f1",
        ascending=False,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "audio_feature_fusion.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFinal fusion comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
