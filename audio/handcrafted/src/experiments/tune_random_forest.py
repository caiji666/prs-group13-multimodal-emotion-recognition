
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.experiments.tuning_utils import (
    load_dataset,
    create_cv,
    evaluate_model,
    save_search_results,
    RANDOM_STATE,
)


def main():

    # 1. Load selected Random Forest dataset
    X_train, y_train, X_val, y_val, groups = load_dataset(
        wavelet="db8",
        normalization="raw",
        frame_ms=30,
    )

    # 2. Define Random Forest classifier
    model = RandomForestClassifier(
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # 3. Hyperparameter search space
    param_distributions = {
        "n_estimators": [
            100, 200, 300, 500
        ],
        "max_depth": [
            5, 10, 15, 20, None
        ],
        "min_samples_split": [
            2, 5, 10, 20
        ],
        "min_samples_leaf": [
            1, 2, 5, 10
        ],
        "max_features": [
            "sqrt", "log2", 0.5, 1.0
        ],
        "class_weight": [
            None, "balanced", "balanced_subsample"
        ],
        "max_samples": [
            0.6, 0.8, 1.0
        ],
    }

    # 4. Actor-independent CV
    cv = create_cv()

    # 5. Randomized hyperparameter search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=60,
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=2,
        random_state=RANDOM_STATE,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    # 6. Tune using training actors only
    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    # 7. Best hyperparameters
    print("\nBest CV Macro-F1:", search.best_score_)
    print("Best hyperparameters:", search.best_params_)

    # 8. Evaluate fitted best model
    best_model = search.best_estimator_

    metrics = evaluate_model(
        best_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 9. Save complete search history
    output_dir = save_search_results(
        search,
        model_name="random_forest",
    )

    # 10. Save best configuration summary
    best_idx = search.best_index_

    summary = {
        "classifier": "Random Forest",
        "wavelet": "db8",
        "normalization": "raw",
        "frame_ms": 30,
        "best_cv_macro_f1": search.best_score_,
        "cv_macro_f1_std": search.cv_results_[
            "std_test_macro_f1"
        ][best_idx],
        "cv_accuracy": search.cv_results_[
            "mean_test_accuracy"
        ][best_idx],
        "cv_train_macro_f1": search.cv_results_[
            "mean_train_macro_f1"
        ][best_idx],
        "best_params": str(search.best_params_),
        **metrics,
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "random_forest_summary.csv",
        index=False,
    )

    # 11. Display top 15 configurations
    results_df = pd.DataFrame(search.cv_results_)

    columns = [
        "rank_test_macro_f1",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
        "param_class_weight",
        "param_max_samples",
        "mean_train_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
    ]

    print("\nTop 15 Random Forest configurations:")

    print(
        results_df[columns]
        .sort_values("rank_test_macro_f1")
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
