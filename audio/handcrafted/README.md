
# PRS — Handcrafted Audio Feature Extraction and Classification

**Project:** Audio-Visual Emotion Recognition through Hybrid Feature Analysis and Multimodal Fusion  
**Programme:** NUS MTech in Artificial Intelligence Systems  
**Module:** Pattern Recognition Systems (PRS)  
**Contribution:** Member 1 (M1) — Handcrafted Audio Features and Audio Feature Fusion

## 1. Overview

This repository implements handcrafted audio feature extraction and classification for speaker-independent emotion recognition using the RAVDESS speech dataset.

The work includes:

- Audio preprocessing and validation
- Time-domain, spectral, and wavelet feature extraction
- Comparison of 30 preprocessing configurations
- Evaluation of nine machine learning classifiers
- Hyperparameter tuning of five shortlisted classifiers
- Comparison of M1 48D, M2 WavLM 128D, and combined 176D audio representations
- Export of aligned M1 features, actor-independent OOF probabilities, and fitted classifier for M5

The repository maintains two implementations:

- **Experimental V2 (main M1 pipeline):** Configurable preprocessing, 30 feature configurations, nine-classifier comparison, tuning, M1–M2 comparison and the M5 handover. All current results come from this pipeline.
- **Baseline V1 (initial prototype):** First 48D pipeline (25 ms frames, db4), MLP/SVM baselines and an early fusion check with M2's first Transformer. Kept for reference; its scripts are in `src/legacy/`.

## 2. Dataset

**Dataset:** RAVDESS — Audio-only speech subset  
**Source:** https://zenodo.org/records/1188976

| Property | Value |
|---|---|
| Recordings | 1,440 |
| Actors | 24 |
| Emotion classes | 8 |
| Sampling rate | 16 kHz |
| Feature dimensions | 48 |

Emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised.

### Actor-Independent Splitting

| Partition | Actors | Recordings |
|---|---:|---:|
| Training | 16 | 960 |
| Validation | 4 | 240 |
| Testing | 4 | 240 |

Recordings from the same actor are not shared across partitions.

The raw RAVDESS recordings are not included in the repository.

## 3. Handcrafted Feature Representation

| Feature Group | Description | Dimensions |
|---|---|---:|
| Time-domain | ZCR, RMS, Energy | 12 |
| Spectral | FFT-based spectral descriptors | 16 |
| Wavelet | DWT coefficient statistics | 20 |
| **Combined** | **Concatenated handcrafted features** | **48** |

Experimental V2 evaluates:

- **Wavelets:** Haar, db2, db4, sym4, db8
- **Normalization:** Raw and normalized
- **Frame lengths:** 20, 25, 30 ms

This produces **30 combined 48D datasets**.

## 4. Repository Structure

```text
M1_Audio_Handcrafted/
├── data/
│   ├── metadata.csv
│   ├── actor_splits.csv
│   ├── audio_splits.csv
│   ├── raw/                         # Local, not tracked
│   └── external/                    # M2-provided exports, not tracked
├── src/
│   ├── dataset.py
│   ├── split_dataset.py
│   ├── preprocessing.py
│   ├── baseline_v1/
│   ├── experimental_v2/
│   ├── experiments/
│   ├── fusion/
│   │   ├── prepare_audio_features.py
│   │   ├── compare_audio_representations.py
│   │   └── tune_audio_representations.py
│   └── handover/
│       └── export_m1_audio.py
├── notebooks/
│   ├── baseline_v1/
│   └── experimental_v2/
├── docs/
├── outputs/
│   ├── features/                    # Generated locally
│   ├── metrics/
│   ├── figures/
│   └── handover/m1_audio/
├── pyproject.toml
├── uv.lock
└── README.md
```

Raw recordings, external M2 artifacts, and generated feature matrices are excluded from Git. The handover directory contains the M1 export artifacts selected for sharing.

## 5. Installation

**Requirements:** Python 3.11 and uv.

From the repository root:

```bash
uv sync --locked
```

Install the locked dependencies before running the pipeline.

## 6. Pipeline Execution

### Dataset Preparation

```bash
uv run --locked python -m src.dataset
uv run --locked python -m src.split_dataset
```

### Baseline V1

```bash
uv run --locked python -m src.baseline_v1.feature_pipeline
```

### Experimental V2

Run sequentially:

```bash
uv run --locked python -m src.experimental_v2.run_preprocessing_validation
uv run --locked python -m src.experimental_v2.extract_time_dataset
uv run --locked python -m src.experimental_v2.extract_spectral_dataset
uv run --locked python -m src.experimental_v2.extract_wavelet_dataset
uv run --locked python -m src.experimental_v2.combine_features
```

### Classifier Comparison

```bash
uv run --locked python -m src.experiments.compare_preprocessing
```

### Hyperparameter Tuning

```bash
uv run --locked python -m src.experiments.tune_logistic_regression
uv run --locked python -m src.experiments.tune_mlp
uv run --locked python -m src.experiments.tune_svc
uv run --locked python -m src.experiments.tune_xgboost
uv run --locked python -m src.experiments.tune_random_forest
```

## 7. Experimental Results

### Initial Classifier Comparison

Nine classifiers were evaluated across 30 preprocessing configurations, producing **270 experiments**.

The five highest-performing classifiers were shortlisted for tuning.

### Default vs Tuned Validation Macro-F1

| Classifier | Default | Tuned |
|---|---:|---:|
| Logistic Regression | **0.3885** | 0.3745 |
| MLP | **0.3692** | 0.3331 |
| XGBoost | **0.3687** | 0.3627 |
| SVC | 0.3413 | **0.3535** |
| Random Forest | 0.3259 | **0.3315** |

### Current Leading Configuration

| Setting | Value |
|---|---|
| Classifier | Logistic Regression (default) |
| Wavelet | db2 |
| Normalization | Normalized |
| Frame length | 20 ms |
| Validation accuracy | 40.00% |
| Validation Macro-F1 | 0.3885 |

This is the highest observed validation Macro-F1 from the current handcrafted classifier comparison.

The validation results are exploratory because the same validation partition was used during preprocessing selection. Final model evaluation remains pending.

## 8. M1–M2 Audio Feature Comparison

The M1 handcrafted representation was aligned with M2's final WavLM export (frozen WavLM Base+ layer 6, trained adapter and attentive statistics pooling; 128D pooled features) using recording identifiers, actor IDs, emotion labels, and split assignments. All 1,440 recordings matched.

The same `StandardScaler` + Logistic Regression pipeline was trained separately on each representation (960 training recordings) and evaluated on the same 240 validation recordings.

| Representation | Dimensions | Default Val Macro-F1 | CV-tuned Val Macro-F1 |
|---|---:|---:|---:|
| Handcrafted (M1) | 48 | 0.3885 | 0.3898 |
| Deep-learned (M2 WavLM) | 128 | **0.7999** | **0.8072** |
| Concatenated (M1 + M2) | 176 | 0.7891 | 0.8000 |

The WavLM features outperformed the handcrafted features for every emotion, and simple feature concatenation did not outperform WavLM features alone. Training-actor CV saturates at 1.0 for the WavLM features because M2's adapter was trained on the same actors, so only the validation scores are informative. These are downstream Logistic Regression experiments and are **not** M2's own classifier results.

An earlier run of this comparison used M2's MFCC Transformer export (deep-only 0.4734, combined 0.4518); it is archived under `archive/mfcc_transformer/`. See [doc 04](docs/04_audio_feature_fusion_comparison.md) for both.

Extract M2's `m5_learned_feature_only.zip` into `data/external/` (pooled CSVs at `data/external/m5_learned_feature_only/pooled/`), then run:

```bash
uv run --locked python -m src.fusion.prepare_audio_features
uv run --locked python -m src.fusion.compare_audio_representations
uv run --locked python -m src.fusion.tune_audio_representations
```

The scripts default to `--m2-source wavlm`; add `--m2-source mfcc_transformer` to regenerate the archived comparison from `data/external/m2_audio/pooled/`. M1's selected combined dataset is read from `outputs/features/experimental_v2/combined/db2/`.

## 9. M5 Audio Handover

The selected M1 handover classifier is the default Logistic Regression pipeline (`C=1.0`) using the 48D db2 / normalized / 20 ms representation.

The export produces partitioned 48D feature CSVs, eight-class probability CSVs, a fitted `logistic_regression.joblib` pipeline, and `metadata.json` under `outputs/handover/m1_audio/`.

- **Training probabilities:** Four-fold actor-independent out-of-fold predictions.
- **Validation probabilities:** Predictions from a model fitted on all 960 training recordings.
- **Test probabilities:** Generated separately and reserved for final evaluation; do not use for fusion-model selection.
- **Alignment:** Use `trial_id` and verify actor, split, and emotion consistency. The canonical M1 label is `surprised` (RAVDESS code 08).

Regenerate the handover after the selected 48D feature dataset is available:

```bash
uv run --locked python -m src.handover.export_m1_audio
```

### Audio Representation for Fusion

The representation comparison (Section 8) selected **WavLM learned-only** features, so the audio input for fusion is **M2's WavLM delivery (`m5_learned_feature_only`)**, which M5 loads directly with `multimodal_fusion/fusion/audio_pilot.py`. No M1 data needs to be sent for the selected configuration.

For convenience, M1 also provides optional ready-to-load files in the `multimodal_fusion` NPZ format under `outputs/handover/final_audio/`:

- **`audio_wavlm.npz`:** M2's WavLM pooled features, probabilities and sequence as a single NPZ (validation Macro-F1 0.8063).
- **`audio_combined.npz`:** handcrafted 48D ‖ WavLM 128D pooled features (176D), M1 Logistic Regression probabilities, and the same sequence (validation Macro-F1 0.7891), for confirming the selection at fusion level.

The NPZ files are not tracked by Git; `metadata.json` documents fields and sources. See [doc 05, Section 12](docs/05_m1_audio_handover.md#12-audio-representation-for-fusion).

```bash
uv run --locked python -m src.handover.export_final_audio
```

## 10. Documentation

| Document | Contents |
|---|---|
| [01 — Handcrafted Feature Pipeline](docs/01_handcrafted_feature_pipeline.md) | Data preparation, preprocessing, feature extraction |
| [02 — Preprocessing and Classifier Comparison](docs/02_preprocessing_and_classifier_comparison.md) | Nine classifiers across 30 configurations |
| [03 — Hyperparameter Tuning Results](docs/03_hyperparameter_tuning_results.md) | Five tuned classifiers and default comparison |
| [04 — Audio Feature Fusion Comparison](docs/04_audio_feature_fusion_comparison.md) | M1 48D vs M2 WavLM 128D vs combined 176D (and archived MFCC Transformer run) |
| [05 — M1 Audio Handover](docs/05_m1_audio_handover.md) | Export schema, OOF probabilities, model and integration guidance |

## 11. Current Status

| Component | Status |
|---|---|
| Actor-independent data preparation and preprocessing | Complete |
| Baseline V1 and Experimental V2 feature extraction | Complete |
| 30 preprocessing configurations and nine-classifier comparison | Complete |
| Five-classifier hyperparameter tuning | Complete |
| M1–M2 recording alignment and audio representation comparison | Complete |
| M1 feature, probability, model and metadata export for M5 | Complete |
| Experimental and handover documentation | Complete |

## 12. Evaluation Notes

The reported validation scores are exploratory: preprocessing and model selection used the designated validation actors. Cross-validation of already exported M2 features is not end-to-end cross-validation of the upstream deep model. The held-out test partition should not be used for further model selection or tuning.
