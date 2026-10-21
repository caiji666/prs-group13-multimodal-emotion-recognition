# Train handcrafted-only SVM and MLP classifiers

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT / "outputs/features/handcrafted_features.csv"
)

METRICS_DIR = PROJECT_ROOT / "outputs/metrics"

RANDOM_SEED = 42


def get_feature_groups(df):
    """Define handcrafted feature-group experiments."""

    feature_cols = [
        c for c in df.columns
        if c not in {
            "sample_id", "trial_id", "actor_id",
            "emotion_label", "split"
        }
    ]

    time_cols = [
        c for c in feature_cols
        if c.startswith(("zcr_", "rms_", "energy_"))
    ]

    spectral_cols = [
        c for c in feature_cols
        if c.startswith("spectral_")
    ]

    dwt_cols = [
        c for c in feature_cols
        if c.startswith("dwt_")
    ]

    reduced_cols = [
        c for c in feature_cols
        if not (
            c.startswith("dwt_")
            and c.endswith("_energy")
        )
    ]

    groups = {
        "time": time_cols,
        "spectral": spectral_cols,
        "dwt": dwt_cols,
        "full": feature_cols,
        "reduced": reduced_cols,
    }

    expected = {
        "time": 12,
        "spectral": 16,
        "dwt": 20,
        "full": 48,
        "reduced": 43,
    }

    for name, cols in groups.items():
        assert len(cols) == expected[name], (
            f"{name}: expected {expected[name]} features, "
            f"got {len(cols)}"
        )

    return groups


def get_models():
    """Define baseline classifiers."""

    return {
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )),
        ]),

        "mlp": Pipeline([
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
        ]),
    }


def main():
    df = pd.read_csv(FEATURE_PATH)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]

    groups = get_feature_groups(df)

    results = []

    for group_name, columns in groups.items():

        X_train = train_df[columns]
        y_train = train_df["emotion_label"]

        X_val = val_df[columns]
        y_val = val_df["emotion_label"]

        for model_name, model in get_models().items():

            print(f"\nTraining {group_name} + {model_name}")

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
                "feature_group": group_name,
                "classifier": model_name,
                "n_features": len(columns),
                "val_accuracy": accuracy,
                "val_macro_f1": macro_f1,
            })

            print(
                f"Accuracy: {accuracy:.4f} | "
                f"Macro-F1: {macro_f1:.4f}"
            )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "val_macro_f1",
        ascending=False,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / "handcrafted_baselines.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFinal validation comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
