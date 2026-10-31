
# Handcrafted Audio Feature Extraction Pipeline

## 1. Overview

This module extracts handcrafted audio features from the RAVDESS speech dataset for speaker-independent emotion recognition.

- **Dataset:** RAVDESS audio-only speech subset
- **Recordings:** 1,440
- **Actors:** 24
- **Emotion classes:** 8
- **Feature representation:** 48-dimensional

## 2. Dataset Partitioning

Actor-independent partitioning prevents recordings from the same speaker appearing across training, validation, and test sets.

| Split | Actors | Recordings |
|---|---:|---:|
| Train | 16 | 960 |
| Validation | 4 | 240 |
| Test | 4 | 240 |

The eight emotion labels are neutral, calm, happy, sad, angry, fearful, disgust, and surprised.

## 3. Audio Preprocessing

Audio preprocessing includes:

- Mono loading and resampling to 16 kHz
- Leading/trailing silence trimming (`top_db=35`)
- Raw and peak-normalized variants
- Frame lengths of 20, 25, and 30 ms

Preprocessing validation successfully processed all 1,440 recordings.

## 4. Handcrafted Feature Representation

| Feature Group | Descriptors | Dimensions |
|---|---|---:|
| Time-domain | ZCR, RMS, Energy | 12 |
| Spectral | Centroid, Bandwidth, Rolloff, Flatness | 16 |
| DWT | A4, D4, D3, D2, D1 statistics | 20 |
| **Combined** | **All three groups** | **48** |

Each descriptor is summarized using four statistical features.

## 5. Experimental Configurations

The experimental pipeline evaluates:

- 5 wavelets: Haar, db2, db4, sym4, db8
- 2 normalization settings: Raw, Normalized
- 3 frame lengths: 20, 25, 30 ms

**Total: 30 combined 48D datasets.**

Each combined dataset contains 1,440 recordings, 48 numerical features, and four metadata columns.

## 6. Pipeline Execution

Download `Audio_Speech_Actors_01-24.zip` from RAVDESS and extract it so the recordings are located at `data/raw/Actor_01/*.wav` to `data/raw/Actor_24/*.wav`.

Run the following modules sequentially from `audio/handcrafted/`:

```bash
uv run --locked python -m src.dataset
uv run --locked python -m src.split_dataset
uv run --locked python -m src.experimental_v2.run_preprocessing_validation
uv run --locked python -m src.experimental_v2.extract_time_dataset
uv run --locked python -m src.experimental_v2.extract_spectral_dataset
uv run --locked python -m src.experimental_v2.extract_wavelet_dataset
uv run --locked python -m src.experimental_v2.combine_features
```

Optional — Baseline V1 (initial prototype) features only:

```bash
uv run --locked python -m src.baseline_v1.feature_pipeline
```

## 7. Generated Outputs

```text
outputs/features/
├── baseline_v1/
│   └── handcrafted_features.csv
└── experimental_v2/
    ├── time/
    ├── spectral/
    ├── dwt/
    └── combined/
```

The experimental feature datasets are used for subsequent preprocessing comparisons, classifier evaluation, and multimodal fusion experiments.
