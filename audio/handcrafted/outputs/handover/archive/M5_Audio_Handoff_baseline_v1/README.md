# Audio Feature Handoff — M1 + M2 to M5

**Project:** PRS — Audio-Visual Emotion Recognition  
**Purpose:** Audio representations and class probabilities for multimodal fusion experiments.

## 1. Handoff Files

```text
M5_Audio_Handoff/
├── config/
│   └── audio_config.json
├── features/
│   ├── handcrafted_features.csv
│   ├── audio_fused_176d.csv
│   └── m2_pooled/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── probabilities/
│   ├── audio_probabilities_train.csv
│   ├── audio_probabilities_val.csv
│   └── audio_probabilities_test.csv
├── results/
│   └── final_test_comparison.csv
├── splits/
│   └── actor_splits.csv
└── README.md
```

## 2. Available Audio Representations

| Representation | Dimensions | File |
|---|---:|---|
| Handcrafted (M1) | 48 | `features/handcrafted_features.csv` |
| Transformer (M2) | 128 | `features/m2_pooled/` |
| Combined (M1 + M2) | 176 | `features/audio_fused_176d.csv` |

**Recommended:** Use the combined 176D representation for feature-level and gated multimodal fusion.

The combined CSV contains **1,440 recordings × 181 columns** (176 features + 5 metadata columns).

## 3. Decision-Level Fusion

Audio class probabilities are provided in `probabilities/`.

| File | Samples | Prediction method |
|---|---:|---|
| `audio_probabilities_train.csv` | 960 | 4-fold actor-grouped out-of-fold predictions |
| `audio_probabilities_val.csv` | 240 | Saved final audio classifier |
| `audio_probabilities_test.csv` | 240 | Saved final audio classifier |

Each file contains five metadata columns and eight probability columns:

`prob_neutral`, `prob_calm`, `prob_happy`, `prob_sad`, `prob_angry`, `prob_fearful`, `prob_disgust`, `prob_surprise`.

These probabilities are generated from the 176D audio fusion MLP.

## 4. Dataset Splits and Alignment

All representations use the shared actor-independent partitions defined in `splits/actor_splits.csv`.

| Split | Actors | Recordings |
|---|---:|---:|
| Train | 16 | 960 |
| Validation | 4 | 240 |
| Test | 4 | 240 |

For audio-visual alignment:

- Use `trial_id` to join corresponding audio and visual recordings.
- Verify matching `actor_id`, `emotion_label` and `split`.
- Do not assume matching CSV row order.
- Do not regenerate actor splits.

## 5. Integration Guidelines

**Feature-level / Gated Fusion:** Use the 176D combined audio representation with the selected visual representation.

**Decision-level Fusion:** Combine the provided audio class probabilities with corresponding visual class probabilities.

**Cross-Attention Fusion:** Coordinate with M2 for sequence-level audio embeddings if temporal cross-attention is required. The provided 128D embeddings are pooled representations.

**Evaluation:**
- Fit feature scalers on training data only.
- Use validation Macro-F1 for model selection.
- Reserve test data for final evaluation.
- Training audio probabilities are actor-grouped out-of-fold predictions to avoid in-sample prediction leakage.

## 6. Standalone Audio Baseline Results

Held-out actor-independent test results:

| Representation | Accuracy | Macro-F1 |
|---|---:|---:|
| Handcrafted (48D) | 35.42% | 0.3500 |
| Transformer (128D) | 54.58% | 0.5151 |
| **Combined (176D)** | **57.50%** | **0.5445** |

Detailed results are available in `results/final_test_comparison.csv`.

## 7. Notes

- All feature exports are precomputed and unscaled.
- The M2 Transformer representations are frozen.
- The audio MLP uses internal early stopping that is not actor-group-aware; the training probability exports use actor-grouped outer folds.
- Raw-audio inference requires the original M1 feature extractor and M2 Transformer encoder.
- The provided audio results serve as unimodal baselines for M5's multimodal fusion experiments.