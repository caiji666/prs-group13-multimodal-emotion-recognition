
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.experiments.tuning_utils import (
    load_dataset,
    create_cv,
    evaluate_model,
    save_search_results,
    RANDOM_STATE,
)


def main():

    # 1. Load selected XGBoost dataset
    X_train, y_train, X_val, y_val, groups = load_dataset(
        wavelet="db8",
        normalization="raw",
        frame_ms=30,
    )

    # 2. Define XGBoost classifier
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    # 3. Hyperparameter search space
    param_distributions = {
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [100, 200, 300, 500],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "reg_lambda": [1, 5, 10, 20],
        "reg_alpha": [0, 0.1, 1, 5],
        "gamma": [0, 0.1, 1],
    }

    # 4. Actor-independent cross-validation
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

    # 8. Evaluate best model
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

    # 9. Save complete search results
    output_dir = save_search_results(
        search,
        model_name="xgboost",
    )

    # 10. Save best configuration summary
    best_idx = search.best_index_

    summary = {
        "classifier": "XGBoost",
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
        output_dir / "xgboost_summary.csv",
        index=False,
    )

    # 11. Display top 15 configurations
    results_df = pd.DataFrame(search.cv_results_)

    columns = [
        "rank_test_macro_f1",
        "param_max_depth",
        "param_min_child_weight",
        "param_learning_rate",
        "param_n_estimators",
        "param_subsample",
        "param_colsample_bytree",
        "param_reg_lambda",
        "param_reg_alpha",
        "param_gamma",
        "mean_train_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
    ]

    print("\nTop 15 XGBoost configurations:")

    print(
        results_df[columns]
        .sort_values("rank_test_macro_f1")
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
