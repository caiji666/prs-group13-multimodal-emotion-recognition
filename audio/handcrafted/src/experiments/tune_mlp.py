
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from src.experiments.tuning_utils import (
    load_dataset,
    create_cv,
    evaluate_model,
    save_search_results,
    RANDOM_STATE,
)


def main():

    # 1. Load selected MLP dataset
    X_train, y_train, X_val, y_val, groups = load_dataset(
        wavelet="sym4",
        normalization="raw",
        frame_ms=20,
    )

    # 2. Define MLP pipeline
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            solver="adam",
            max_iter=1000,
            tol=1e-4,
            n_iter_no_change=30,
            early_stopping=False,
            random_state=RANDOM_STATE,
        ),
    )

    # 3. Hyperparameter search space
    
    param_distributions = {
        "mlpclassifier__hidden_layer_sizes": [
            (16,),
            (24,),
            (32,),
            (48,),
            (64,),
            (96,),
            (128,),
            (24, 12),
            (48, 24),
            (64, 32),
            (96, 48),
            (128, 64),
            (64, 32, 16),
            (128, 64, 32),
        ],

        "mlpclassifier__alpha": [
            0.001,
            0.01,
            0.1,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ],

        "mlpclassifier__learning_rate_init": [
            0.0001,
            0.0005,
            0.001,
            0.002,
        ],

        "mlpclassifier__activation": [
            "relu",
            "tanh",
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
        cv=create_cv(),
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
        model_name="mlp_round2",
    )

    # 10. Save model summary
    best_idx = search.best_index_

    summary = {
        "classifier": "MLP",
        "wavelet": "sym4",
        "normalization": "raw",
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
        output_dir / "mlp_round2_summary.csv",
        index=False,
    )

    # 11. Display top 15 configurations
    results_df = pd.DataFrame(search.cv_results_)

    columns = [
        "rank_test_macro_f1",
        "param_mlpclassifier__hidden_layer_sizes",
        "param_mlpclassifier__alpha",
        "param_mlpclassifier__learning_rate_init",
        "param_mlpclassifier__activation",
        "mean_train_macro_f1",
        "mean_test_macro_f1",
        "std_test_macro_f1",
        "mean_test_accuracy",
    ]

    print("\nTop 15 MLP hyperparameter configurations:")

    print(
        results_df[columns]
        .sort_values("rank_test_macro_f1")
        .head(15)
        .to_string(index=False)
    )

    # 12. Inspect MLP convergence
    mlp = best_model.named_steps["mlpclassifier"]

    print("\nBest MLP training information:")
    print("Iterations:", mlp.n_iter_)
    print("Final training loss:", mlp.loss_)
    print("Reached iteration limit:", mlp.n_iter_ >= mlp.max_iter)

    print("\nSaved:", output_dir)


if __name__ == "__main__":
    main()
