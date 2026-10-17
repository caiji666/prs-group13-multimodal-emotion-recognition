
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from feature_fusion import load_aligned_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs/models/handcrafted_transformer.joblib"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/probabilities"
)

CLASS_NAMES = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprise",
]

CLASS_LABELS = np.arange(8)
N_FOLDS = 4


def make_probability_frame(df, probabilities, split):
    """Build a labeled probability table."""

    result = pd.DataFrame({
        "sample_id": df["sample_id_m1"].to_numpy(),
        "trial_id": df["trial_id"].to_numpy(),
        "actor_id": df["actor_id_m1"].to_numpy(),
        "emotion_label": df["emotion_label"].to_numpy(),
        "split": split,
    })

    for i, emotion in enumerate(CLASS_NAMES):
        result[f"prob_{emotion}"] = probabilities[:, i]

    return result


def main():
    merged, _ = load_aligned_features()

    artifact = joblib.load(MODEL_PATH)

    final_model = artifact["model"]
    feature_cols = artifact["feature_columns"]

    train_df = merged[
        merged["split_m1"] == "train"
    ].copy()

    val_df = merged[
        merged["split_m1"] == "val"
    ].copy()

    test_df = merged[
        merged["split_m1"] == "test"
    ].copy()

    X_train = train_df[feature_cols]
    y_train = train_df["label_m1"].astype(int).to_numpy()
    groups = train_df["actor_id_m1"].to_numpy()

    assert len(train_df) == 960
    assert len(val_df) == 240
    assert len(test_df) == 240

    # Out-of-fold training probabilities
    oof_probabilities = np.zeros(
        (len(train_df), len(CLASS_LABELS)),
        dtype=np.float64,
    )

    assigned = np.zeros(len(train_df), dtype=bool)

    cv = GroupKFold(n_splits=N_FOLDS)

    for fold, (fit_idx, holdout_idx) in enumerate(
        cv.split(X_train, y_train, groups),
        start=1,
    ):
        print(f"Training OOF fold {fold}/{N_FOLDS}")

        model = clone(final_model)

        model.fit(
            X_train.iloc[fit_idx],
            y_train[fit_idx],
        )

        fold_probabilities = model.predict_proba(
            X_train.iloc[holdout_idx]
        )

        fold_classes = model.named_steps[
            "classifier"
        ].classes_

        # Align probabilities to canonical class order
        for j, label in enumerate(fold_classes):
            oof_probabilities[
                holdout_idx, int(label)
            ] = fold_probabilities[:, j]

        assigned[holdout_idx] = True

    assert assigned.all()

    # Validation/test predictions from saved final model
    def final_probabilities(df):
        probabilities = final_model.predict_proba(
            df[feature_cols]
        )

        aligned = np.zeros(
            (len(df), len(CLASS_LABELS)),
            dtype=np.float64,
        )

        model_classes = final_model.named_steps[
            "classifier"
        ].classes_

        for j, label in enumerate(model_classes):
            aligned[:, int(label)] = probabilities[:, j]

        return aligned

    val_probabilities = final_probabilities(val_df)
    test_probabilities = final_probabilities(test_df)

    # Validate probability matrices
    for name, probabilities in [
        ("train", oof_probabilities),
        ("val", val_probabilities),
        ("test", test_probabilities),
    ]:
        assert np.isfinite(probabilities).all()
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

        np.testing.assert_allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-6,
        )

        print(
            f"{name:5s} | "
            f"Shape: {probabilities.shape}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": make_probability_frame(
            train_df, oof_probabilities, "train"
        ),
        "val": make_probability_frame(
            val_df, val_probabilities, "val"
        ),
        "test": make_probability_frame(
            test_df, test_probabilities, "test"
        ),
    }

    for split, df in outputs.items():
        output_path = (
            OUTPUT_DIR / f"audio_probabilities_{split}.csv"
        )

        df.to_csv(output_path, index=False)

        print("Saved:", output_path)

    print("\nAudio probability export completed!")


if __name__ == "__main__":
    main()
