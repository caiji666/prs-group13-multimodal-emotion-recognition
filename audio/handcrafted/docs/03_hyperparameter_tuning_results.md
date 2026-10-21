
# Hyperparameter Tuning and Classifier Comparison

## 1. Objective

Optimize the five shortlisted classifiers and compare their performance against the original default configurations using their respective selected 48D handcrafted datasets.

## 2. Tuning Methodology

- **Cross-validation:** 4-fold StratifiedGroupKFold
- **Grouping variable:** actor_id
- **Primary tuning metric:** Mean CV Macro-F1
- **Secondary metric:** Accuracy
- **Training recordings:** 960
- **Validation recordings:** 240

GridSearchCV was used for Logistic Regression and SVC. RandomizedSearchCV was used for MLP, XGBoost, and Random Forest.

The best CV-selected configuration of each classifier was refitted on the complete training partition before validation evaluation.

## 3. Best Tuned Hyperparameters

| Classifier | Selected Hyperparameters |
|---|---|
| Logistic Regression | C=0.1, saga, l1_ratio=0.25, class_weight=None |
| MLP (Round 2) | (64,), ReLU, alpha=2.0, learning_rate=0.002 |
| SVC | RBF, C=10, gamma=0.001, class_weight=balanced |
| XGBoost | max_depth=2, learning_rate=0.05, n_estimators=500, reg_lambda=10, reg_alpha=1 |
| Random Forest | n_estimators=500, max_depth=20, min_samples_leaf=2, max_samples=0.6 |

Full configurations are available in the tuning result CSVs.

## 4. Default vs Tuned Validation Performance

| Classifier | Default Macro-F1 | Tuned Macro-F1 | Change |
|---|---:|---:|---:|
| Logistic Regression | **0.3885** | 0.3745 | -0.0140 |
| MLP | **0.3692** | 0.3331 | -0.0361 |
| XGBoost | **0.3687** | 0.3627 | -0.0059 |
| SVC | 0.3413 | **0.3535** | +0.0122 |
| Random Forest | 0.3259 | **0.3315** | +0.0056 |

Both default and tuned classifiers were fitted on the complete training partition and evaluated using the same designated validation actors.

## 5. Tuned Model Generalization

| Classifier | Train Macro-F1 | Val Macro-F1 | F1 Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.4705 | 0.3745 | 0.0960 |
| SVC | 0.4784 | 0.3535 | 0.1248 |
| MLP | 0.6622 | 0.3331 | 0.3291 |
| XGBoost | 0.8074 | 0.3627 | 0.4447 |
| Random Forest | 0.9781 | 0.3315 | 0.6466 |

## 6. Key Findings

- Default Logistic Regression achieved the highest observed validation Macro-F1 (0.3885).
- Hyperparameter tuning improved validation Macro-F1 for SVC and Random Forest.
- Logistic Regression and SVC exhibited smaller training–validation gaps than the other tuned classifiers.
- MLP, XGBoost, and Random Forest continued to show substantial overfitting.
- Higher model complexity did not consistently improve speaker-independent validation performance.

## 7. Current Model Candidate

**Classifier:** Default Logistic Regression

**Preprocessing:**
- Wavelet: db2
- Normalization: Normalized
- Frame length: 20 ms

**Validation results:**
- Accuracy: 0.4000
- Macro-F1: 0.3885

This is the current validation-leading candidate for the handcrafted audio representation, pending final model selection and evaluation.

## 8. Limitations and Next Steps

Preprocessing configurations were shortlisted using the designated validation partition. Consequently, validation performance is exploratory and may be affected by selection bias.

The test partition remains reserved for final evaluation.

Next steps include:

1. Finalize the handcrafted representation and classifier.
2. Conduct feature ablation experiments.
3. Compare handcrafted, deep-learned, and concatenated audio representations.
4. Prepare aligned features and classifier probabilities for multimodal fusion.

## 9. Result Files

```text
outputs/metrics/experimental_v2/tuning/
├── logistic_regression_search_results.csv
├── logistic_regression_summary.csv
├── mlp_search_results.csv
├── mlp_summary.csv
├── mlp_round2_search_results.csv
├── mlp_round2_summary.csv
├── svc_search_results.csv
├── svc_summary.csv
├── xgboost_search_results.csv
├── xgboost_summary.csv
├── random_forest_search_results.csv
└── random_forest_summary.csv
```
