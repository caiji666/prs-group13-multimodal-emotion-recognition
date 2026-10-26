
# Audio Feature Representation and Fusion Comparison

## 1. Objective

Evaluate the effectiveness of handcrafted, deep-learned, and concatenated audio feature representations for speaker-independent emotion recognition using the RAVDESS speech dataset.

The experiment investigates whether combining M1's handcrafted features with M2's deep-learned features improves classification performance.

## 2. Feature Representations

| Representation | Source | Dimensions |
|---|---|---:|
| Handcrafted | M1 — Time, Spectral, DWT | 48 |
| Deep-learned | M2 — frozen WavLM Base+ with trained adapter and attentive statistics pooling | 128 |
| Combined | Concatenation of M1 and M2 | 176 |

**M1 configuration:** db2 wavelet, normalized audio, 20 ms frame length.

**M2 configuration:** M2's final Round 05 WavLM export (`m5_learned_feature_only`): frozen WavLM Base+ layer 6, temporal adapter and attentive statistics pooling, giving 128D pooled features (`F_audio_learned_feature_only_000`–`127`). M2's own MLP head on these features reaches validation Macro-F1 0.8063 (`M2_encoder_wavlm_s42_20261020_05`).

The combined representation is constructed through feature concatenation:

\[
F_{\text{combined}} = [F_{\text{handcrafted}} \; || \; F_{\text{deep}}]
\]

### M2 Input Files

Extract `m5_learned_feature_only.zip` (provided by M2) into `data/external/` so the pooled CSVs are located at:

```text
data/external/m5_learned_feature_only/pooled/
├── train.csv
├── val.csv
└── test.csv
```

Only the 128 `F_audio_learned_feature_only_*` columns are used. M2's `P_audio_learned_feature_only_*` probability columns are ignored.

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
├── learned_128d.csv           # WavLM
├── combined_176d.csv          # M1 + WavLM
└── archive/mfcc_transformer/  # earlier M2 export (Section 10)
    ├── learned_128d.csv
    └── combined_176d.csv
```

Each dataset contains five metadata columns and its corresponding numerical feature representation. M2 features are renamed `m2_f_000`–`m2_f_127`.

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
| **Deep-learned 128D** | 1.0000 | **0.8083** | **0.7999** | 0.2001 |
| Combined 176D | 1.0000 | 0.7958 | 0.7891 | 0.2109 |

The WavLM deep-learned 128D representation achieved the highest validation Macro-F1, 0.4114 above the handcrafted representation.

Logistic Regression on the WavLM features performed almost as well as M2's own MLP head on the same features (0.7999 vs 0.8063), indicating that the exported features are already close to linearly separable by emotion.

Concatenating handcrafted and WavLM features reduced validation Macro-F1 by 0.0108 compared with WavLM features alone.

### Per-Class Validation F1

| Emotion | Handcrafted 48D | Deep-learned 128D | Combined 176D |
|---|---:|---:|---:|
| neutral | 0.3077 | **0.6875** | **0.6875** |
| calm | 0.5556 | **0.8267** | 0.7949 |
| happy | 0.3922 | **0.7200** | **0.7200** |
| sad | 0.2195 | **0.6761** | 0.6571 |
| angry | 0.5610 | **0.9538** | **0.9538** |
| fearful | 0.3333 | **0.7692** | 0.7500 |
| disgust | 0.4000 | **0.8136** | **0.8136** |
| surprised | 0.3390 | **0.9524** | 0.9355 |

The WavLM features scored higher than the handcrafted features for every emotion. Concatenation matched WavLM alone for four emotions and was slightly lower for the other four.

## 6. Tuned Logistic Regression Results

The regularization parameter `C` was optimized separately for each representation using actor-independent cross-validation.

| Representation | Best C | CV Macro-F1 | Val Accuracy | Val Macro-F1 |
|---|---:|---:|---:|---:|
| Handcrafted 48D | 0.10 | 0.3392 | 0.4083 | 0.3898 |
| Deep-learned 128D | 0.001 | 1.0000 | **0.8125** | **0.8072** |
| Combined 176D | 0.001 | 1.0000 | 0.8042 | 0.8000 |

**CV saturation:** for the deep-learned and combined representations, mean CV Macro-F1 was 1.0000 for every `C` from 0.001 to 100 (0.8587 at `C=0.0001`). GridSearchCV breaks the tie by selecting the first value, 0.001. M2's adapter was trained on the same 16 training actors, so their learned features are perfectly separable within training-actor CV. Cross-validation on the training actors therefore cannot select `C` for learned features; the validation comparison is the meaningful one.

### Default vs Tuned Validation Macro-F1

| Representation | Default | Tuned | Change |
|---|---:|---:|---:|
| Handcrafted 48D | 0.3885 | **0.3898** | +0.0013 |
| Deep-learned 128D | 0.7999 | **0.8072** | +0.0073 |
| Combined 176D | 0.7891 | **0.8000** | +0.0109 |

The strongest observed validation performance was **deep-learned 128D with tuned Logistic Regression** (Macro-F1 0.8072). The combined representation remained below deep-learned features alone in both experiments.

## 7. Generalization Analysis

| Representation | Tuned Train Macro-F1 | Tuned Val Macro-F1 | F1 Gap |
|---|---:|---:|---:|
| Handcrafted 48D | 0.4817 | 0.3898 | **0.0920** |
| Deep-learned 128D | 1.0000 | 0.8072 | 0.1928 |
| Combined 176D | 1.0000 | 0.8000 | 0.2000 |

Training Macro-F1 is 1.0000 for both representations containing WavLM features, because M2's feature extractor was fitted on the same training actors. For learned features, the training score is therefore not an indication of overfitting by the Logistic Regression; only the validation actors provide an unbiased comparison.

### Error Overlap

Validation predictions of the default handcrafted and deep-learned classifiers:

| Outcome | Recordings |
|---|---:|
| Both correct | 87 |
| Handcrafted correct only | 9 |
| Deep-learned correct only | 107 |
| Both incorrect | 37 |

An oracle choosing the correct prediction of the two would reach 84.6% accuracy, compared with 80.8% for the deep-learned classifier. The handcrafted features provide only limited complementary information, which simple concatenation with Logistic Regression did not exploit.

## 8. Key Findings

1. **WavLM deep-learned features outperform handcrafted features:** 0.7999 vs 0.3885 validation Macro-F1 with default Logistic Regression, and higher F1 for every emotion.

2. **Feature concatenation did not improve the strongest validation result:** The combined 176D representation scored below deep-learned features alone with both default (0.7891 vs 0.7999) and tuned (0.8000 vs 0.8072) Logistic Regression.

3. **A linear classifier is sufficient for the WavLM features:** Logistic Regression reached almost the same validation Macro-F1 as M2's MLP head (0.7999 vs 0.8063).

4. **Complementarity is limited:** The handcrafted classifier was correct on only 9 of the 240 validation recordings that the deep-learned classifier misclassified.

5. **Fusion effectiveness is classifier-dependent:** These results evaluate feature concatenation using Logistic Regression and do not establish the maximum achievable performance of the combined representation.

## 9. Limitations

- M1's preprocessing configuration was selected through earlier validation experiments.
- M2's deep representation originated from a supervised model trained and selected separately. M2 selected its checkpoint using the same validation actors, so validation scores of the learned features are optimistic.
- WavLM Base+ is pretrained on external speech data; only M2's adapter and head were trained on RAVDESS.
- Cross-validation on pre-extracted deep features does not constitute end-to-end cross-validation of the M2 feature extractor, and it saturates at 1.0 for the WavLM features (Section 6).
- Validation results are exploratory and may be influenced by earlier model-selection decisions.
- The held-out test partition remains unused for final performance reporting.

M2's own classifier results should be reported separately from the Logistic Regression classifiers trained on exported pooled features.

## 10. Earlier Comparison with MFCC Transformer Features (Archived)

Before the WavLM handoff, this comparison was run on M2's earlier Stage 3 **MFCC Transformer** export (epoch 14; M2 validation Macro-F1 0.4508). An earlier version of this document labelled these results as WavLM; they were produced from the MFCC Transformer export.

| Representation | Default Val Macro-F1 | Tuned C | CV Macro-F1 | Tuned Val Macro-F1 |
|---|---:|---:|---:|---:|
| Handcrafted 48D | 0.3885 | 0.10 | 0.3392 | 0.3898 |
| Deep-learned 128D (MFCC Transformer) | **0.4734** | 0.01 | 0.6670 | 0.4262 |
| Combined 176D | 0.4518 | 0.05 | 0.6662 | **0.4492** |

The same conclusion held: concatenation did not outperform the default deep-only classifier. With these weaker learned features, the handcrafted features scored higher for disgust (0.4000 vs 0.2745) and fearful (0.3333 vs 0.2979); this advantage disappears with the WavLM features.

These results are kept under `archive/mfcc_transformer/` and can be regenerated with `--m2-source mfcc_transformer`, which expects the earlier export in `data/external/m2_audio/pooled/`.

## 11. Result Files

```text
outputs/metrics/audio_fusion/
├── representation_comparison.csv
├── per_class_val_f1.csv
├── tuning/
│   ├── handcrafted_48d_search.csv
│   ├── deep_learned_128d_search.csv
│   ├── combined_176d_search.csv
│   └── tuned_representation_comparison.csv
└── archive/mfcc_transformer/      # same files for the earlier M2 export
```

### Reproduction Commands

Run from `audio/handcrafted/` after extracting the M2 export (Section 2):

```bash
uv run --locked python -m src.fusion.prepare_audio_features
uv run --locked python -m src.fusion.compare_audio_representations
uv run --locked python -m src.fusion.tune_audio_representations
```

The scripts default to `--m2-source wavlm`. Add `--m2-source mfcc_transformer` to all three commands to regenerate the archived comparison.

## 12. Conclusion

The WavLM deep-learned 128D representation achieved the highest observed validation Macro-F1 (0.7999 with default and 0.8072 with tuned Logistic Regression), far above the handcrafted 48D representation (0.3885).

Concatenating handcrafted and WavLM features did not improve on WavLM features alone, and the two representations share most of their errors.

These findings complete the M1 audio representation comparison and provide aligned feature datasets for subsequent multimodal integration.
