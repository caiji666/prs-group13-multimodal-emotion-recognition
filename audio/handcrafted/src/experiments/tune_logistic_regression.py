
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from src.experiments.tuning_utils import (
    load_dataset,
    create_cv,
    evaluate_model,
    save_search_results,
    RANDOM_STATE,
)


def main():

    # 1. Load selected dataset
    X_train, y_train, X_val, y_val, groups = load_dataset(
        wavelet="db2",
        normalization="norm",
        frame_ms=20,
    )

    # 2. Define classifier pipeline
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
    )

    # 3. Hyperparameter search space
    
    param_grid = [
        # L2 regularization
        {
            "logisticregression__solver": [
                "lbfgs",
                "newton-cg",
            ],
            "logisticregression__C": [
                0.001, 0.01, 0.05, 0.1,
                0.5, 1.0, 2.0, 5.0,
                10.0, 100.0,
            ],
            "logisticregression__l1_ratio": [0.0],
            "logisticregression__class_weight": [
                None, "balanced"
            ],
        },

        # L1 and Elastic Net regularization
        {
            "logisticregression__solver": ["saga"],
            "logisticregression__C": [
                0.01, 0.1, 0.5, 1.0, 5.0, 10.0,
            ],
            "logisticregression__l1_ratio": [
                0.25, 0.5, 0.75, 1.0,
            ],
            "logisticregression__class_weight": [
                None, "balanced"
            ],
        },
    ]


    # 4. Actor-independent cross-validation
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
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    # 6. Train and tune using training actors only
    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    # 7. Best hyperparameters
    print("\nBest CV Macro-F1:", search.best_score_)
    print("Best hyperparameters:", search.best_params_)

    # 8. Evaluate on designated validation actors
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
        model_name="logistic_regression",
    )

    # 10. Save best configuration summary
    best_idx = search.best_index_

    summary = {
        "classifier": "Logistic Regression",
        "wavelet": "db2",
        "normalization": "norm",
        "frame_ms": 20,
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
        output_dir / "logistic_regression_summary.csv",
        index=False,
    )

    # 11. Display top configurations
    results_df = pd.DataFrame(search.cv_results_)

    columns = [
        "rank_test_macro_f1",
        "param_logisticregression__C",
        "mean_train_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
    ]

    print("\nLogistic Regression hyperparameter comparison:")

    print(
        results_df[columns]
        .sort_values("rank_test_macro_f1")
        .to_string(index=False)
    )

    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
