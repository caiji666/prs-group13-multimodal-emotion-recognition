
# Audio Feature Representation and Fusion Comparison

## 1. Objective

Evaluate the effectiveness of handcrafted, deep-learned, and concatenated audio feature representations for speaker-independent emotion recognition using the RAVDESS speech dataset.

The experiment investigates whether combining M1's handcrafted features with M2's deep-learned features improves classification performance.

## 2. Feature Representations

| Representation | Source | Dimensions |
|---|---|---:|
| Handcrafted | M1 — Time, Spectral, DWT | 48 |
| Deep-learned | M2 — WavLM-based pooled features | 128 |
| Combined | Concatenation of M1 and M2 | 176 |

**M1 configuration:** db2 wavelet, normalized audio, 20 ms frame length.

**M2 configuration:** Round 05 WavLM-based deep audio model with exported 128D pooled features.

The combined representation is constructed through feature concatenation:

\[
F_{\text{combined}} = [F_{\text{handcrafted}} \; || \; F_{\text{deep}}]
\]

## 3. Dataset Alignment

M1 and M2 feature datasets were aligned using a shared recording identifier.

- M1's seven-field `sample_id` was converted to a six-field `trial_id` by removing the modality prefix.
- M2's six-field `sample_id` was matched against M1's `trial_id`.
- Actor IDs, emotion labels, and dataset splits were validated.
- All 1,440 recordings were successfully aligned without mismatches.

### Dataset Partitioning

| Split | Actors | Recordings |
|---|---:|---:|
| Training | 16 | 960 |
| Validation | 4 | 240 |
| Testing | 4 | 240 |
| **Total** | **24** | **1,440** |

The actor-independent partitions were preserved throughout the experiments.

### Generated Datasets

```text
outputs/features/audio_fusion/
├── handcrafted_48d.csv
├── learned_128d.csv
└── combined_176d.csv
```

Each dataset contains five metadata columns and its corresponding numerical feature representation.

## 4. Experimental Methodology

The same Logistic Regression classification pipeline was evaluated across all three representations.

**Classifier configuration:**
- StandardScaler
- LogisticRegression with L2 regularization
- Solver: lbfgs
- Random state: 42

Two experiments were performed:

1. **Default comparison:** Logistic Regression with `C=1.0`.
2. **Regularization tuning:** GridSearchCV over 11 `C` values using four-fold actor-independent StratifiedGroupKFold.

Hyperparameters were selected using mean CV Macro-F1. The selected models were refitted on all 960 training recordings and evaluated on the 240 validation recordings.

**Primary metric:** Validation Macro-F1  
**Secondary metric:** Validation Accuracy

The test partition was excluded from model fitting, hyperparameter selection, and validation comparison.

## 5. Default Logistic Regression Results

| Representation | Train Macro-F1 | Val Accuracy | Val Macro-F1 | F1 Gap |
|---|---:|---:|---:|---:|
| Handcrafted 48D | 0.5084 | 0.4000 | 0.3885 | 0.1199 |
| **Deep-learned 128D** | 0.8814 | **0.4792** | **0.4734** | 0.4080 |
| Combined 176D | 0.9561 | 0.4708 | 0.4518 | 0.5043 |

The deep-learned 128D representation achieved the highest validation Macro-F1 under default Logistic Regression.

Concatenating handcrafted and deep-learned features increased training performance but did not improve validation performance over deep features alone.

## 6. Tuned Logistic Regression Results

The regularization parameter `C` was optimized separately for each representation using actor-independent cross-validation.

| Representation | Best C | CV Macro-F1 | Val Accuracy | Val Macro-F1 |
|---|---:|---:|---:|---:|
| Handcrafted 48D | 0.10 | 0.3392 | 0.4083 | 0.3898 |
| Deep-learned 128D | 0.01 | **0.6670** | 0.4458 | 0.4262 |
| Combined 176D | 0.05 | 0.6662 | **0.4750** | **0.4492** |

The combined representation achieved the highest validation Macro-F1 among the three CV-tuned classifiers, although the deep-only representation achieved a slightly higher mean CV Macro-F1.

### Default vs Tuned Validation Macro-F1

| Representation | Default | Tuned | Change |
|---|---:|---:|---:|
| Handcrafted 48D | 0.3885 | **0.3898** | +0.0013 |
| Deep-learned 128D | **0.4734** | 0.4262 | -0.0472 |
| Combined 176D | **0.4518** | 0.4492 | -0.0026 |

The strongest observed validation performance across both experiments remained **deep-learned 128D with default Logistic Regression**, achieving Macro-F1 of 0.4734.

## 7. Generalization Analysis

| Representation | Tuned Train Macro-F1 | Tuned Val Macro-F1 | F1 Gap |
|---|---:|---:|---:|
| Handcrafted 48D | 0.4817 | 0.3898 | **0.0920** |
| Deep-learned 128D | 0.7308 | 0.4262 | 0.3047 |
| Combined 176D | 0.8311 | 0.4492 | 0.3820 |

The combined representation exhibited a larger training–validation gap than either individual representation under the tuned classifier.

This suggests that concatenation increased the classifier's ability to fit training data without producing a consistent improvement on unseen validation actors.

## 8. Key Findings

1. **Deep-learned features outperform handcrafted features:** The 128D representation achieved higher validation Macro-F1 than the 48D representation in both default and tuned experiments.

2. **Feature concatenation did not improve the strongest validation result:** The combined 176D representation did not outperform the default deep-only classifier.

3. **Regularization affected representations differently:** Stronger regularization slightly improved handcrafted validation performance but reduced deep-only and combined validation performance.

4. **Overfitting increased with feature complexity:** The combined representation achieved higher training Macro-F1 but also exhibited a larger generalization gap.

5. **Fusion effectiveness is classifier-dependent:** These results evaluate feature concatenation using Logistic Regression and do not establish the maximum achievable performance of the combined representation.

## 9. Limitations

- M1's preprocessing configuration was selected through earlier validation experiments.
- M2's deep representation originated from a supervised model trained and selected separately.
- Cross-validation on pre-extracted deep features does not constitute end-to-end cross-validation of the M2 feature extractor.
- Validation results are exploratory and may be influenced by earlier model-selection decisions.
- The held-out test partition remains unused for final performance reporting.

M2's original WavLM-based classifier results should be reported separately from the Logistic Regression classifiers trained on exported pooled features.

## 10. Result Files

```text
outputs/metrics/audio_fusion/
├── representation_comparison.csv
└── tuning/
    ├── handcrafted_48d_search.csv
    ├── deep_learned_128d_search.csv
    ├── combined_176d_search.csv
    └── tuned_representation_comparison.csv
```

### Reproduction Commands

```bash
uv run --locked python -m src.fusion.prepare_audio_features
uv run --locked python -m src.fusion.compare_audio_representations
uv run --locked python -m src.fusion.tune_audio_representations
```

## 11. Conclusion

The deep-learned 128D representation achieved the highest observed validation Macro-F1 (0.4734) using default Logistic Regression.

Although concatenation improved training performance, the combined 176D representation did not consistently outperform deep features alone.

These findings establish the baseline for future audio feature fusion experiments and provide aligned feature datasets for subsequent multimodal integration.
