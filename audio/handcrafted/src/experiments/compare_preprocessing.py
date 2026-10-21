
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

MODELS = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
    ),

    "SVC": make_pipeline(
        StandardScaler(),
        SVC(),
    ),

    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(),
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),

    "Gaussian Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_jobs=1,
    ),

    "AdaBoost": AdaBoostClassifier(
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
        random_state=42,
        n_jobs=1,
        eval_metric="mlogloss",
    ),

    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(
            max_iter=400,
            random_state=42,
        ),
    ),
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = (
    PROJECT_ROOT / "outputs/features/experimental_v2/combined"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/metrics/experimental_v2"
)

WAVELETS = ("haar", "db2", "db4", "sym4", "db8")
FRAME_LENGTHS = (20, 25, 30)

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def evaluate_configuration(
    model_name: str,
    model,
    wavelet: str,
    normalize: bool,
    frame_ms: int,
) -> dict:
    """Train one classifier on one preprocessing configuration."""

    norm_label = "norm" if normalize else "raw"

    # 1. Load dataset
    file_path = (
        FEATURE_DIR
        / wavelet
        / f"combined_{norm_label}_{frame_ms}ms.csv"
    )

    dataset = pd.read_csv(file_path)

    # 2. Select 48 numerical features
    feature_columns = [
        col for col in dataset.columns
        if col not in METADATA_COLUMNS
    ]

    # 3. Separate training and validation
    train_df = dataset[dataset["split"] == "train"]
    val_df = dataset[dataset["split"] == "val"]

    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]

    # Encode emotion labels consistently
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(
        train_df["emotion_label"]
    )

    y_val = encoder.transform(
        val_df["emotion_label"]
    )

    # Basic validation
    assert len(feature_columns) == 48
    assert len(train_df) == 960
    assert len(val_df) == 240

    assert set(train_df["actor_id"]).isdisjoint(
        set(val_df["actor_id"])
    )

    assert np.isfinite(X_train.to_numpy()).all()
    assert np.isfinite(X_val.to_numpy()).all()

    # 4. Create fresh classifier
    classifier = clone(model)

    # 5. Train
    classifier.fit(X_train, y_train)

    # 6. Predict
    train_pred = classifier.predict(X_train)
    train_accuracy = accuracy_score(
        y_train,
        train_pred,
    )
    val_pred = classifier.predict(X_val)

    # 7. Calculate metrics
    train_macro_f1 = f1_score(
        y_train,
        train_pred,
        average="macro",
        zero_division=0,
    )

    val_accuracy = accuracy_score(
        y_val,
        val_pred,
    )

    val_macro_f1 = f1_score(
        y_val,
        val_pred,
        average="macro",
        zero_division=0,
    )
    
    f1_gap = train_macro_f1 - val_macro_f1

    return {
        "classifier": model_name,
        "wavelet": wavelet,
        "normalization": norm_label,
        "frame_ms": frame_ms,
        "n_features": len(feature_columns),
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "f1_gap": f1_gap,
    }



if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Evaluate all 9 classifiers
    for model_name, model in MODELS.items():

        print(f"\n{'=' * 60}")
        print(f"Training classifier: {model_name}")
        print(f"{'=' * 60}")

        # Evaluate all 30 preprocessing configurations
        for wavelet in WAVELETS:
            for normalize in (False, True):
                for frame_ms in FRAME_LENGTHS:

                    result = evaluate_configuration(
                        model_name=model_name,
                        model=model,
                        wavelet=wavelet,
                        normalize=normalize,
                        frame_ms=frame_ms,
                    )

                    results.append(result)
                    pd.DataFrame(results).to_csv(
                        OUTPUT_DIR / "all_classifier_preprocessing_comparison.csv",
                        index=False,
                    )

                    print(
                        f"{result['wavelet']} | "
                        f"{result['normalization']} | "
                        f"{result['frame_ms']}ms | "
                        f"Train F1: {result['train_macro_f1']:.4f} | "
                        f"Val F1: {result['val_macro_f1']:.4f}"
                    )

    # Complete comparison results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["val_macro_f1", "f1_gap"],
        ascending=[False, True],
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_DIR / "all_classifier_preprocessing_comparison.csv",
        index=False,
    )

    # Best preprocessing configuration for each classifier
    best_per_model = (
        results_df
        .sort_values("val_macro_f1", ascending=False)
        .drop_duplicates(subset="classifier", keep="first")
        .reset_index(drop=True)
    )

    best_path = (
        OUTPUT_DIR / "best_preprocessing_per_classifier.csv"
    )

    best_per_model.to_csv(best_path, index=False)

    print("\n" + "=" * 80)
    print("BEST PREPROCESSING CONFIGURATION PER CLASSIFIER")
    print("=" * 80)

    print(
        best_per_model[
            [
                "classifier",
                "wavelet",
                "normalization",
                "frame_ms",
                "train_accuracy",
                "train_macro_f1",
                "val_accuracy",
                "val_macro_f1",
                "f1_gap",
            ]
        ].to_string(index=False)
    )

    # Overall highest validation Macro-F1
    best_overall = results_df.iloc[0]

    print("\nBEST OVERALL VALIDATION CONFIGURATION")
    print("Classifier:", best_overall["classifier"])
    print("Wavelet:", best_overall["wavelet"])
    print("Normalization:", best_overall["normalization"])
    print("Frame length:", best_overall["frame_ms"], "ms")
    print("Validation Accuracy:", round(best_overall["val_accuracy"], 4))
    print("Validation Macro-F1:", round(best_overall["val_macro_f1"], 4))

    print("\nSaved:", best_path)
