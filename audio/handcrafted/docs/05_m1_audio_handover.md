
# M1 Audio Features and Classifier Handover

## 1. Overview

This document describes the handcrafted audio features, classifier, and probability outputs provided by Member 1 (M1) for multimodal fusion experiments conducted by Member 5 (M5).

**Dataset:** RAVDESS audio-only speech subset  
**Recordings:** 1,440  
**Emotion classes:** 8  
**Handcrafted representation:** 48D  
**Classifier:** Logistic Regression

The exported features and probabilities use actor-independent training, validation, and test partitions aligned with the group project.

## 2. Selected Model Configuration

The selected M1 classifier is the default Logistic Regression model, which achieved the highest observed validation Macro-F1 during the handcrafted classifier comparison.

| Parameter | Value |
|---|---|
| Classifier | Logistic Regression |
| Regularization | L2 |
| C | 1.0 |
| Solver | lbfgs |
| Max iterations | 3000 |
| Random state | 42 |
| Feature scaling | StandardScaler |
| Feature dimensions | 48 |
| Wavelet | db2 |
| Audio normalization | Enabled |
| Frame length | 20 ms |

**Observed validation performance:**

- Validation Accuracy: **0.4000**
- Validation Macro-F1: **0.3885**

These results are from the designated validation partition, not the final test set.

## 3. Handover Files

The export script generates the following directory:

```text
outputs/handover/m1_audio/
├── features/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── probabilities/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── model/
│   └── logistic_regression.joblib
└── metadata.json
```

### Feature Datasets

| File | Recordings | Features |
|---|---:|---:|
| features/train.csv | 960 | 48 |
| features/val.csv | 240 | 48 |
| features/test.csv | 240 | 48 |

Each CSV contains five metadata columns and 48 numerical handcrafted features.

### Probability Datasets

| File | Recordings | Probability Columns |
|---|---:|---:|
| probabilities/train.csv | 960 | 8 |
| probabilities/val.csv | 240 | 8 |
| probabilities/test.csv | 240 | 8 |

Each probability CSV contains five metadata columns and eight class probabilities.

## 4. Metadata and Recording Alignment

The following metadata columns are included in the exported CSVs:

| Column | Description |
|---|---|
| sample_id | Original seven-field RAVDESS audio filename identifier |
| trial_id | Six-field identifier shared across modalities |
| actor_id | RAVDESS actor identifier |
| emotion_label | Ground-truth emotion |
| split | train, val, or test |

The `trial_id` is obtained by removing the modality prefix (`03-`) from M1's audio-only `sample_id`.

**M5 should use `trial_id` for cross-modality alignment** and verify that actor IDs, labels, and splits match before fusion.

## 5. Emotion Classes and Probability Schema

The canonical emotion order is:

| Class Index | Emotion | Probability Column |
|---:|---|---|
| 0 | neutral | p_neutral |
| 1 | calm | p_calm |
| 2 | happy | p_happy |
| 3 | sad | p_sad |
| 4 | angry | p_angry |
| 5 | fearful | p_fearful |
| 6 | disgust | p_disgust |
| 7 | surprised | p_surprised |

Each probability represents the classifier's estimated likelihood for the corresponding emotion.

All probabilities are finite, lie between 0 and 1, and sum to approximately 1 for each recording.

**Important:** M1 uses `surprised`, while some other project modules use `surprise`. These refer to the same RAVDESS emotion code `08` and must be mapped consistently during fusion.

## 6. Probability Generation

### Training Partition

Training probabilities are generated using **4-fold actor-independent out-of-fold (OOF) prediction** with `StratifiedGroupKFold`.

- Grouping variable: `actor_id`
- Number of folds: 4
- Each fold uses 12 training actors and 4 held-out actors.
- Each training recording receives probabilities from a model trained without its actor.
- All 960 training recordings receive exactly one OOF prediction.

This reduces in-sample prediction leakage when training a downstream fusion classifier.

### Validation and Test Partitions

A final Logistic Regression pipeline is fitted using all 960 training recordings.

The fitted model generates probabilities for:

- 240 validation recordings
- 240 test recordings

Neither partition is used to fit the M1 classifier.

The test predictions must remain reserved for the group's final evaluation.

## 7. Model and Metadata Files

### logistic_regression.joblib

Contains the fitted classification pipeline:

```python
StandardScaler()
LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=3000,
    random_state=42,
)
```

The model expects the same 48 handcrafted features in the original training-column order.

### metadata.json

Contains:

- Selected model configuration
- Handcrafted preprocessing settings
- Feature names and ordering
- Canonical emotion labels
- Probability column names
- Probability-generation methodology
- Dataset partition sizes

M5 should use this metadata to validate the handover schema.

## 8. Reproducing the Handover

From `audio/handcrafted/`:

```bash
uv run --locked python -m src.handover.export_m1_audio
```

The script exports the 48D features, generates actor-independent OOF training probabilities, fits the final classifier, and saves validation/test probabilities, the model, and metadata.

## 9. Integration Guidelines for M5

- Use `trial_id` to align M1 audio outputs with other modalities.
- Use the eight probability columns for decision-level fusion.
- Use the 48D feature columns for feature-level fusion if required.
- Preserve the original train/validation/test partitions.
- Train fusion models using OOF training probabilities rather than in-sample predictions.
- Apply the same emotion-class order across modalities.
- Keep the test partition untouched until the final fusion configuration is selected.

For decision-level fusion, other modalities should ideally provide similarly out-of-fold training predictions to avoid stacking leakage.

## 10. Limitations

- The handcrafted preprocessing configuration was selected using earlier validation experiments.
- Validation results are exploratory and may be affected by model-selection bias.
- OOF predictions are generated using fixed handcrafted features; the audio feature extraction itself is not retrained within each fold.
- The exported Logistic Regression model is specific to the selected 48D representation.
- Final test-set evaluation is pending.

## 11. Handover Status

| Deliverable | Status |
|---|---|
| Handcrafted 48D features | Complete |
| Actor-independent dataset alignment | Verified |
| Selected Logistic Regression classifier | Complete |
| OOF training probabilities | Complete |
| Validation/test probabilities | Complete |
| Fitted model and metadata | Complete |
| Audio representation selection for fusion (Section 12) | Complete |
| Optional ready-to-load NPZ files (Section 12) | Complete |
| M5 multimodal fusion evaluation | Pending |

The M1 audio outputs are ready for integration into the group's multimodal fusion pipeline.

## 12. Audio Representation for Fusion

Sections 1–11 describe the M1-only handcrafted handover. This section records which audio representation the fusion module should use, and describes optional files that package it in the `multimodal_fusion` NPZ format (`fusion.data.load_modality`).

### Selection

The audio representation comparison ([doc 04](04_audio_feature_fusion_comparison.md)) selected the **WavLM learned-only** representation: validation Macro-F1 0.7999, compared with 0.7891 for the handcrafted + WavLM concatenation and 0.3885 for handcrafted features alone. Concatenating the handcrafted features does not improve on the learned features, so no M1 features are added to the audio input for fusion.

**The selected audio input is M2's WavLM Round 05 delivery (`m5_learned_feature_only`).** It already contains everything the fusion module needs: pooled features, M2's probabilities and the sequence for cross-attention. M5 loads it directly with `multimodal_fusion/fusion/audio_pilot.py` (`import_archive`), so M1 does not need to send any audio data for the selected configuration.

### Optional Ready-to-Load Files

For convenience, `src/handover/export_final_audio.py` also packages the audio input as single NPZ files in the fusion contract. M5 may use these instead of converting M2's per-sample delivery, but they are not required:

| File | Pooled | Probabilities | Val Accuracy | Val Macro-F1 |
|---|---|---|---:|---:|
| `audio_wavlm.npz` | M2 WavLM 128D | M2 WavLM Round 05 MLP head | 0.8125 | 0.8063 |
| `audio_combined.npz` | M1 48D ‖ M2 WavLM 128D = 176D | M1 Logistic Regression on 176D | 0.7958 | 0.7891 |

`audio_wavlm.npz` contains exactly M2's data (pooled, probabilities, sequence) in the single-file format. `audio_combined.npz` is only needed if M5 wants to confirm the selection at fusion level. Both files contain the same `sequence` and `lengths`: M2's `F_sequence` [T, 128], unchanged and zero-padded at the end to T = 215 (valid lengths 58–215).

### Files

```text
outputs/handover/final_audio/
├── audio_wavlm.npz      # optional: M2 WavLM delivery as one NPZ (not tracked)
├── audio_combined.npz   # optional: handcrafted + WavLM variant (not tracked)
└── metadata.json        # fields, sources, column layout, validation scores
```

| Field | Shape | Description |
|---|---|---|
| `sample_ids` | [1440] | Six-field trial IDs |
| `labels` | [1440] int64 | RAVDESS emotion code − 1 |
| `actors` | [1440] int64 | Actor ID |
| `class_names` | [8] | neutral, calm, happy, sad, angry, fearful, disgust, surprise |
| `pooled` | [1440, D] float32 | D = 128 (WavLM) or 176 (combined) |
| `probs` | [1440, 8] float32 | Class probabilities |
| `sequence` | [1440, 215, 128] float32 | M2 WavLM sequence for cross-attention |
| `lengths` | [1440] int64 | Valid sequence steps |

Class 7 uses the fusion spelling `surprise`.

### Probability Generation

- **`audio_wavlm.npz`:** M2's softmax probabilities for all splits. Training probabilities are in-sample.
- **`audio_combined.npz`:** `StandardScaler` + Logistic Regression (`C=1.0`); 4-fold actor-independent OOF probabilities for training and a full-training model for validation and test. Because M2's adapter was trained on the training actors, these OOF probabilities remain optimistic.

Training-set audio probabilities are therefore over-confident in both files, as they are in M2's delivery. Tune decision-fusion weights on the validation partition. Test probabilities were generated but not scored.

### Usage (Optional Files)

```bash
python -m fusion validate --audio <path>/audio_wavlm.npz --visual <path>/visual.npz --split configs/actors_m2.json
```

Both files were checked with `fusion.data.load_modality`, `split_indices` (with `configs/actors_m2.json`), `load_pair`, `fit_normalizers` and `normalize`.

### Reproducing the Export

Requires `outputs/features/audio_fusion/handcrafted_48d.csv` and M2's `m5_learned_feature_only.zip` extracted into `data/external/`:

```bash
uv run --locked python -m src.handover.export_final_audio
```
