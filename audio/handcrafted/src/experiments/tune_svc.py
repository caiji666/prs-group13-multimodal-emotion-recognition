
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from src.experiments.tuning_utils import (
    load_dataset,
    create_cv,
    evaluate_model,
    save_search_results,
)


def main():

    # 1. Load selected SVC dataset
    X_train, y_train, X_val, y_val, groups = load_dataset(
        wavelet="sym4",
        normalization="norm",
        frame_ms=25,
    )

    # 2. Define classifier pipeline
    model = make_pipeline(
        StandardScaler(),
        SVC(),
    )

    # 3. Hyperparameter search space
    param_grid = [
        # Linear SVC
        {
            "svc__kernel": ["linear"],
            "svc__C": [
                0.01, 0.1, 1.0, 10.0, 100.0, 300.0
            ],
            "svc__class_weight": [
                None, "balanced"
            ],
        },

        # RBF SVC
        {
            "svc__kernel": ["rbf"],
            "svc__C": [
                0.01, 0.1, 1.0, 10.0, 100.0, 300.0
            ],
            "svc__gamma": [
                "scale", 0.0001, 0.001, 0.01, 0.1
            ],
            "svc__class_weight": [
                None, "balanced"
            ],
        },
    ]

    # 4. Actor-independent CV
    cv = create_cv()

    # 5. Grid search
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
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

    # 9. Save search results
    output_dir = save_search_results(
        search,
        model_name="svc",
    )

    # 10. Save summary
    best_idx = search.best_index_

    summary = {
        "classifier": "SVC",
        "wavelet": "sym4",
        "normalization": "norm",
        "frame_ms": 25,
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
        output_dir / "svc_summary.csv",
        index=False,
    )

    # 11. Display top 15 configurations
    results_df = pd.DataFrame(search.cv_results_)

    columns = [
        "rank_test_macro_f1",
        "param_svc__kernel",
        "param_svc__C",
        "param_svc__gamma",
        "param_svc__class_weight",
        "mean_train_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
    ]

    print("\nTop 15 SVC hyperparameter configurations:")

    print(
        results_df[columns]
        .sort_values("rank_test_macro_f1")
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
