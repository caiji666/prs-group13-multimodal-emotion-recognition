
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from feature_fusion import load_aligned_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "outputs/models"
METRICS_DIR = PROJECT_ROOT / "outputs/metrics"
FIGURE_DIR = PROJECT_ROOT / "outputs/figures"

MODEL_NAMES = [
    "handcrafted_only",
    "transformer_only",
    "handcrafted_transformer",
]

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

LABELS = list(range(8))

EXPECTED_TEST_ACTORS = {2, 3, 8, 17}


def evaluate_model(name, test_df):
    """Evaluate a saved model on held-out test recordings."""

    artifact_path = MODEL_DIR / f"{name}.joblib"

    artifact = joblib.load(artifact_path)

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    assert artifact["representation"] == name

    assert set(feature_columns).issubset(test_df.columns)

    X_test = test_df[feature_columns]
    y_test = test_df["label_m1"].astype(int)

    assert np.isfinite(X_test.to_numpy()).all()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        labels=LABELS,
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        labels=LABELS,
        zero_division=0,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=LABELS,
    )

    # Save classification report
    report_df = pd.DataFrame(report).T
    report_df.to_csv(
        METRICS_DIR / f"{name}_classification_report.csv"
    )

    # Save confusion matrix values
    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        METRICS_DIR / f"{name}_confusion_matrix.csv"
    )

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(9, 8))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        colorbar=False,
    )

    ax.set_title(f"Test Confusion Matrix — {name}")

    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{name}_confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Save individual predictions
    predictions_df = pd.DataFrame({
        "sample_id": test_df["sample_id_m1"].values,
        "actor_id": test_df["actor_id_m1"].values,
        "true_label": y_test.values,
        "predicted_label": predictions,
    })

    predictions_df.to_csv(
        METRICS_DIR / f"{name}_test_predictions.csv",
        index=False,
    )

    return {
        "model": name,
        "n_features": len(feature_columns),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
    }


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    merged, _ = load_aligned_features()

    test_df = merged[
        merged["split_m1"] == "test"
    ].copy()

    assert len(test_df) == 240
    assert set(test_df["actor_id_m1"]) == EXPECTED_TEST_ACTORS

    print("Test recordings:", len(test_df))
    print("Test actors:", sorted(EXPECTED_TEST_ACTORS))

    results = []

    for name in MODEL_NAMES:
        print(f"\nEvaluating: {name}")

        result = evaluate_model(name, test_df)
        results.append(result)

        print(
            f"Accuracy: {result['test_accuracy']:.4f} | "
            f"Macro-F1: {result['test_macro_f1']:.4f} | "
            f"Weighted-F1: {result['test_weighted_f1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    output_path = METRICS_DIR / "final_test_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print("\nFinal held-out test comparison:")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
