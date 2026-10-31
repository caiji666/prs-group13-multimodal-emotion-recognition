
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs/features/audio_fusion/handcrafted_48d.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/handover/m1_audio"
)

FEATURE_DIR = OUTPUT_DIR / "features"
PROB_DIR = OUTPUT_DIR / "probabilities"
MODEL_DIR = OUTPUT_DIR / "model"

RANDOM_STATE = 42

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
    label: index
    for index, label in enumerate(EMOTIONS)
}

METADATA_COLUMNS = [
    "sample_id",
    "trial_id",
    "actor_id",
    "emotion_label",
    "split",
]


def create_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
    )


def export_probabilities(
    df,
    probabilities,
    output_path,
):
    probability_columns = [
        f"p_{emotion}"
        for emotion in EMOTIONS
    ]

    assert probabilities.shape == (
        len(df), len(EMOTIONS)
    )

    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
    )

    output = df[METADATA_COLUMNS].copy()

    output[probability_columns] = probabilities

    output.to_csv(output_path, index=False)


def main():

    dataset = pd.read_csv(DATASET_PATH)

    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    assert len(dataset) == 1440
    assert len(feature_columns) == 48
    assert dataset["sample_id"].is_unique
    assert dataset["trial_id"].is_unique

    assert np.isfinite(
        dataset[feature_columns].to_numpy()
    ).all()

    splits = {
        split: dataset[
            dataset["split"] == split
        ].copy()
        for split in ["train", "val", "test"]
    }

    assert len(splits["train"]) == 960
    assert len(splits["val"]) == 240
    assert len(splits["test"]) == 240

    # Verify actor independence across all partitions
    actor_sets = {
        split: set(df["actor_id"])
        for split, df in splits.items()
    }

    assert actor_sets["train"].isdisjoint(
        actor_sets["val"]
    )
    assert actor_sets["train"].isdisjoint(
        actor_sets["test"]
    )
    assert actor_sets["val"].isdisjoint(
        actor_sets["test"]
    )

    for directory in [
        FEATURE_DIR,
        PROB_DIR,
        MODEL_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # Export 48D features for each partition
    for split, df in splits.items():

        output_path = (
            FEATURE_DIR / f"{split}.csv"
        )

        df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"Features: {split} | "
            f"{len(df)} recordings"
        )

    # Training data
    train_df = splits["train"]

    X_train = train_df[feature_columns]

    y_train = train_df[
        "emotion_label"
    ].map(LABEL_MAP).to_numpy()

    groups = train_df[
        "actor_id"
    ].to_numpy()

    assert np.isfinite(y_train).all()

    # Actor-independent OOF predictions
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.zeros(
        (len(train_df), len(EMOTIONS)),
        dtype=np.float64,
    )

    oof_counts = np.zeros(
        len(train_df),
        dtype=np.int32,
    )

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups),
        start=1,
    ):

        train_actors = set(groups[train_idx])
        val_actors = set(groups[val_idx])

        assert train_actors.isdisjoint(val_actors)

        model = create_model()

        model.fit(
            X_train.iloc[train_idx],
            y_train[train_idx],
        )

        fold_probabilities = model.predict_proba(
            X_train.iloc[val_idx]
        )

        # Align probabilities to canonical emotion order
        aligned_probabilities = np.zeros(
            (len(val_idx), len(EMOTIONS))
        )

        for index, class_id in enumerate(
            model.classes_
        ):
            aligned_probabilities[:, class_id] = (
                fold_probabilities[:, index]
            )

        oof_probabilities[val_idx] = (
            aligned_probabilities
        )

        oof_counts[val_idx] += 1

        print(
            f"OOF fold {fold}: "
            f"{len(train_idx)} training, "
            f"{len(val_idx)} held-out recordings"
        )

    assert np.all(oof_counts == 1)

    export_probabilities(
        train_df,
        oof_probabilities,
        PROB_DIR / "train.csv",
    )

    # Fit final model on all training recordings
    final_model = create_model()

    final_model.fit(
        X_train,
        y_train,
    )

    assert np.array_equal(
        final_model.classes_,
        np.arange(len(EMOTIONS)),
    )

    # Generate validation and test probabilities
    for split in ["val", "test"]:

        df = splits[split]

        probabilities = final_model.predict_proba(
            df[feature_columns]
        )

        export_probabilities(
            df,
            probabilities,
            PROB_DIR / f"{split}.csv",
        )

        print(
            f"Probabilities: {split} | "
            f"{len(df)} recordings"
        )

    # Save fitted model
    model_path = (
        MODEL_DIR / "logistic_regression.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
    )

    # Save metadata
    metadata = {
        "model": "LogisticRegression",
        "model_parameters": {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 3000,
            "random_state": RANDOM_STATE,
        },
        "preprocessing": {
            "wavelet": "db2",
            "normalization": "norm",
            "frame_ms": 20,
        },
        "feature_dimensions": 48,
        "feature_columns": feature_columns,
        "emotion_labels": EMOTIONS,
        "probability_columns": [
            f"p_{emotion}"
            for emotion in EMOTIONS
        ],
        "train_probability_method": (
            "4-fold actor-independent OOF"
        ),
        "validation_probability_method": (
            "Full-training fitted model"
        ),
        "test_probability_method": (
            "Full-training fitted model"
        ),
        "split_counts": {
            split: len(df)
            for split, df in splits.items()
        },
    }

    metadata_path = (
        OUTPUT_DIR / "metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("\nM1 audio handover completed!")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
