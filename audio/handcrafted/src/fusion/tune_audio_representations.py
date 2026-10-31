
import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
)
from sklearn.metrics import accuracy_score, f1_score

from src.fusion.m2_sources import (
    HANDCRAFTED_PATH,
    feature_dir,
    metrics_dir,
    parse_source,
)


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

C_VALUES = [
    0.0001,
    0.001,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    100.0,
]


def evaluate_representation(name, dataset_path, output_dir):

    dataset = pd.read_csv(dataset_path)

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

    assert len(train_df) == 960
    assert len(val_df) == 240
    assert y_train.notna().all()
    assert y_val.notna().all()

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    # Same classifier architecture for all representations
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        ),
    )

    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid={
            "logisticregression__C": C_VALUES
        },
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=2,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train.astype(int),
        groups=groups,
    )

    best_model = search.best_estimator_

    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)

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

    # Save complete CV search history
    filename_prefix = (
        name.lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    search_df = pd.DataFrame(search.cv_results_)

    search_df.to_csv(
        output_dir / f"{filename_prefix}_search.csv",
        index=False,
    )

    return {
        "representation": name,
        "n_features": len(feature_columns),
        "best_C": search.best_params_[
            "logisticregression__C"
        ],
        "best_cv_macro_f1": search.best_score_,
        "cv_macro_f1_std": search.cv_results_[
            "std_test_macro_f1"
        ][search.best_index_],
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

    source = parse_source(
        "Tune Logistic Regression C for each audio representation."
    )

    datasets = {
        "Handcrafted 48D": HANDCRAFTED_PATH,
        "Deep-learned 128D": feature_dir(source) / "learned_128d.csv",
        "Combined 176D": feature_dir(source) / "combined_176d.csv",
    }

    output_dir = metrics_dir(source) / "tuning"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("M2 source:", source)

    results = []

    for name, dataset_path in datasets.items():

        print(f"\nTuning: {name}")

        result = evaluate_representation(
            name,
            dataset_path,
            output_dir,
        )

        results.append(result)

        print(
            f"Best C: {result['best_C']} | "
            f"CV F1: {result['best_cv_macro_f1']:.4f} | "
            f"Val F1: {result['val_macro_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "val_macro_f1",
        ascending=False,
    ).reset_index(drop=True)

    output_path = (
        output_dir / "tuned_representation_comparison.csv"
    )

    results_df.to_csv(output_path, index=False)

    print("\nTuned audio representation comparison:")
    print(results_df.round(4).to_string(index=False))

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
