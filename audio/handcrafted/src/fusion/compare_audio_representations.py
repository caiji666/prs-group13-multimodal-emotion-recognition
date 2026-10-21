
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT / "outputs/features/audio_fusion"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/metrics/audio_fusion"
)

DATASETS = {
    "Handcrafted 48D": "handcrafted_48d.csv",
    "Deep-learned 128D": "learned_128d.csv",
    "Combined 176D": "combined_176d.csv",
}

METADATA_COLUMNS = [
    "sample_id",
    "trial_id",
    "actor_id",
    "emotion_label",
    "split",
]

EMOTIONS = [
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
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


def evaluate_representation(name, filename):

    dataset = pd.read_csv(FEATURE_DIR / filename)

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

    assert y_train.notna().all()
    assert y_val.notna().all()

    assert len(train_df) == 960
    assert len(val_df) == 240

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

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
        "representation": name,
        "n_features": len(feature_columns),
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


def main():

    results = []

    for name, filename in DATASETS.items():

        result = evaluate_representation(
            name,
            filename,
        )

        results.append(result)

        print(
            f"{name:20s} | "
            f"Val Accuracy: {result['val_accuracy']:.4f} | "
            f"Val Macro-F1: {result['val_macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "val_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR / "representation_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\nAudio representation comparison:")
    print(results_df.round(4).to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
