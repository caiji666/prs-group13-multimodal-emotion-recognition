
# Preprocessing and Classifier Comparison

## 1. Objective

Evaluate the effects of preprocessing configurations and classifier selection on speaker-independent emotion recognition using 48D handcrafted audio features.

## 2. Experimental Setup

- **Preprocessing configurations:** 30
- **Classifiers:** 9
- **Total experiments:** 270
- **Training recordings:** 960
- **Validation recordings:** 240
- **Primary metric:** Validation Macro-F1
- **Secondary metric:** Validation Accuracy

### Classifiers Evaluated

Logistic Regression, SVC, KNN, Decision Tree, Gaussian Naive Bayes, Random Forest, AdaBoost, XGBoost, and MLP.

The initial comparison used default or near-default classifier configurations without systematic hyperparameter optimization.

## 3. Best Preprocessing Configuration per Classifier

| Classifier | Wavelet | Normalization | Frame (ms) | Val Macro-F1 |
|---|---|---|---:|---:|
| Logistic Regression | db2 | norm | 20 | **0.3885** |
| MLP | sym4 | raw | 20 | 0.3692 |
| XGBoost | db8 | raw | 30 | 0.3687 |
| SVC | sym4 | norm | 25 | 0.3413 |
| Random Forest | db8 | raw | 30 | 0.3259 |
| Gaussian Naive Bayes | db8 | raw | 20 | 0.3194 |
| AdaBoost | db2 | raw | 25 | 0.3124 |
| Decision Tree | db2 | raw | 25 | 0.2938 |
| KNN | db2 | raw | 20 | 0.2631 |

## 4. Key Findings

- Logistic Regression achieved the highest validation Macro-F1 (0.3885).
- The best preprocessing configuration varied across classifiers.
- MLP, XGBoost, and Random Forest showed substantial training–validation gaps.
- The five highest-performing classifiers were shortlisted for hyperparameter tuning.

## 5. Generated Results

```text
outputs/metrics/experimental_v2/
├── all_classifier_preprocessing_comparison.csv
└── best_preprocessing_per_classifier.csv
```

These results represent exploratory validation comparisons. The test partition was not used for model selection.
