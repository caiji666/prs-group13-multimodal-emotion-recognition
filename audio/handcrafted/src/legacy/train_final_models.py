
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from src.legacy.feature_fusion import (
    load_aligned_features,
    get_feature_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"

RANDOM_SEED = 42

METADATA_COLS = {
    "sample_id", "trial_id", "actor_id",
    "emotion_label", "split", "m2_sample_id",
    "label",
}


def create_model():
    """Reproduce the selected Day 4 MLP configuration."""

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

    fusion_groups = get_feature_groups(merged, m1)

    handcrafted_cols = [
        c for c in m1.columns
        if c not in METADATA_COLS
    ]

    groups = {
        "handcrafted_only": handcrafted_cols,
        "transformer_only": fusion_groups["transformer_only"],
        "handcrafted_transformer": fusion_groups[
            "handcrafted_transformer"
        ],
    }

    train_df = merged[merged["split_m1"] == "train"]

    y_train = train_df["label_m1"].astype(int)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for name, columns in groups.items():
        print(f"\nTraining final model: {name}")

        X_train = train_df[columns]

        assert np.isfinite(X_train.to_numpy()).all()

        model = create_model()
        model.fit(X_train, y_train)

        artifact = {
            "model": model,
            "feature_columns": columns,
            "representation": name,
            "random_seed": RANDOM_SEED,
            "label_mapping": {
                0: "neutral",
                1: "calm",
                2: "happy",
                3: "sad",
                4: "angry",
                5: "fearful",
                6: "disgust",
                7: "surprise",
            },
        }

        output_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(artifact, output_path)

        print(f"Saved: {output_path}")
        print(f"Input features: {len(columns)}")

    print("\nAll selected models saved successfully!")


if __name__ == "__main__":
    main()
