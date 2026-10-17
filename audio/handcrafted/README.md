# PRS — Handcrafted Audio Features and Deep Audio Feature Fusion

**Group project:** Multimodal Emotion Recognition Using Speech and Facial Expressions  
**Programme:** NUS MTech in Artificial Intelligence Systems  
**Module:** Pattern Recognition Systems (PRS)  
**Contribution:** Member 1 (M1) — Handcrafted Audio Feature Extraction, Classification, and Audio Feature Fusion

## 1. Overview

This repository implements the audio component developed by Member 1 for the PRS group project. It studies eight-class speech emotion recognition on the RAVDESS speech subset using handcrafted signal-processing features and their combination with deep-learned audio embeddings supplied by Member 2 (M2).

**M1 primary responsibility:** prepare the audio data; extract time-domain, spectral, and wavelet features; compare feature groups using SVM and MLP classifiers; and evaluate the resulting handcrafted representation.

**M1 secondary responsibility:** align M1 features with M2's pooled Transformer embeddings, train downstream audio-fusion classifiers, compare handcrafted-only, Transformer-only, and combined representations, and export aligned audio features for the group's subsequent audio-visual fusion work.

**Best held-out audio result:** a 176-dimensional handcrafted + Transformer representation with an MLP achieved **57.50% accuracy and 0.5445 Macro-F1** on 240 recordings from four held-out actors. These are results for *audio-only classification*, not the group's final audio-visual system.

## 2. Dataset and partitioning

**Dataset:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)  
**Source:** https://zenodo.org/records/1188976  
**Subset:** `Audio_Speech_Actors_01-24.zip` (audio-only speech)

- 1,440 WAV recordings from 24 actors (60 recordings per actor).
- Eight emotion labels: neutral, calm, happy, sad, angry, fearful, disgust, surprise.
- Original audio sampling rate: 48 kHz; analysis sampling rate: 16 kHz.
- Dataset filenames encode modality, vocal channel, emotion, intensity, statement, repetition, and actor.

`src/dataset.py` creates `data/metadata.csv`. `src/split_dataset.py` generates the actor-independent partition files. The same actor assignments were shared across the team and verified against M2's exported embeddings.

| Partition | Actors | Recordings | Actor IDs |
|---|---:|---:|---|
| Train | 16 | 960 | 01, 04, 05, 06, 07, 10, 11, 12, 13, 15, 16, 18, 19, 20, 23, 24 |
| Validation | 4 | 240 | 09, 14, 21, 22 |
| Test | 4 | 240 | 02, 03, 08, 17 |

The partitioning was created using seed 42. **No actor appears in more than one partition.** Validation was used to compare candidate models; the held-out test partition was used for final evaluation.

The dataset is not committed to Git. Download it from the source above and extract the actor directories under `data/raw/`, for example `data/raw/Actor_01/*.wav`.

## 3. Environment and setup

**Python:** 3.11  
**Dependency management:** [`uv`](https://docs.astral.sh/uv/) via `pyproject.toml` and `uv.lock`

From the repository root:

```bash
uv sync --locked
```

Run scripts with the locked environment:

```bash
uv run --locked python src/dataset.py
uv run --locked python src/split_dataset.py
uv run --locked python src/preprocessing.py
uv run --locked python src/time_features.py
uv run --locked python src/spectral_features.py
uv run --locked python src/wavelet_features.py
uv run --locked python src/feature_pipeline.py
```

The individual preprocessing and feature scripts include single-recording demonstrations. `feature_pipeline.py` performs batch feature extraction.

For the M2 integration steps, place the received pooled CSV exports at:

```text
data/external/m2_audio/
├── audio_config.json
└── pooled/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

M2 also supplied matching `.npz` files; the integration scripts here use the CSV files. External M2 files are excluded from Git and must be obtained separately from M2.

## 4. Repository structure

```text
M1_Audio_Handcrafted/
├── data/
│   ├── actor_splits.csv
│   ├── audio_splits.csv
│   ├── metadata.csv
│   ├── raw/                         # local RAVDESS WAVs (ignored)
│   └── external/m2_audio/           # M2 embeddings (ignored)
├── notebooks/                       # preprocessing, feature, and result analysis
├── src/
│   ├── dataset.py
│   ├── split_dataset.py
│   ├── preprocessing.py
│   ├── time_features.py
│   ├── spectral_features.py
│   ├── wavelet_features.py
│   ├── feature_pipeline.py
│   ├── validate_m2_features.py
│   ├── train_classifier.py
│   ├── feature_fusion.py
│   ├── train_final_models.py
│   ├── evaluate_final_models.py
│   └── export_audio_features.py
├── outputs/
│   ├── features/                    # generated features (ignored)
│   │   └── export/                  # 176D handoff tables (ignored)
│   ├── models/                      # trained .joblib artifacts (ignored)
│   ├── metrics/                     # evaluation CSVs
│   ├── figures/                     # confusion matrices
│   └── preprocessing_validation.csv
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

Generated directories may not appear in a fresh clone until their corresponding scripts are run.

## 5. Audio preprocessing

Implemented in `src/preprocessing.py`:

1. Load audio as mono.
2. Resample from 48 kHz to 16 kHz.
3. Trim leading/trailing silence with `top_db=35`.
4. Support optional peak normalization; **normalization is disabled** in the reported handcrafted feature experiments to preserve amplitude-related information.

All 1,440 recordings passed the preprocessing validation.

| Measure | Result |
|---|---:|
| Successful / failed | 1,440 / 0 |
| Mean original duration | 3.70 s |
| Mean processed duration | 2.01 s |
| Mean duration retained | 53.95% |
| Minimum duration retained | 38.56% |

Detailed checks are recorded in `outputs/preprocessing_validation.csv`. The trimming configuration and amplitude-normalization choice should remain consistent when reproducing the handcrafted features.

## 6. Handcrafted audio representation (48D)

### Time-domain features — 12 dimensions

`src/time_features.py` computes frame-level zero-crossing rate (ZCR), root mean square (RMS), and frame energy. Each is summarized by mean, standard deviation, minimum, and maximum: **3 × 4 = 12** features.

### Spectral features — 16 dimensions

`src/spectral_features.py` applies a Hann window and real FFT to each frame, extracting spectral centroid, spectral bandwidth, spectral rolloff (85% cumulative **power**), and windowed FFT spectral energy. The same four aggregation statistics yield **4 × 4 = 16** features.

| Parameter | Value |
|---|---|
| Sample rate | 16,000 Hz |
| Frame length | 400 samples / 25 ms |
| Hop length | 160 samples / 10 ms |
| Frame overlap | 60% |
| Spectral window | Hann |

### Discrete Wavelet Transform — 20 dimensions

`src/wavelet_features.py` applies the `db4` wavelet at decomposition level 4. For each coefficient group **A4, D4, D3, D2, D1**, it calculates mean, variance, mean squared coefficient energy, and Shannon entropy of normalized squared coefficients: **5 × 4 = 20** features.

Training-set analysis found that DWT variance and mean squared energy were nearly perfectly correlated at each level. Their identity, `mean(c²) = variance(c) + mean(c)²`, was verified numerically (maximum differences around 10⁻¹⁶). A reduced 43D representation excludes the five DWT energy columns for comparison; the full 48D representation is retained as the primary handcrafted export.

### Batch feature extraction

`src/feature_pipeline.py` combines all three feature groups and saves:

- `outputs/features/time_spectral_features.csv` — earlier 28D time + spectral representation (from the Day 2 run).
- `outputs/features/handcrafted_features.csv` — complete **1,440 × 48** handcrafted feature matrix, plus five metadata columns.

The final 48D table was validated: 1,440 unique samples, no missing/non-finite feature values, and the expected 960/240/240 split distribution.

## 7. Handcrafted classification experiments

`src/train_classifier.py` compares five representations with SVM (RBF) and MLP classifiers. Each pipeline fits `StandardScaler` on the **training partition only**. The primary selection metric is validation **Macro-F1**.

| Representation | Dimensions | Best classifier | Validation Macro-F1 |
|---|---:|---|---:|
| Time only | 12 | SVM | 0.3334 |
| Spectral only | 16 | SVM | 0.2699 |
| DWT only | 20 | MLP | 0.2429 |
| **Full handcrafted** | **48** | **MLP** | **0.3518** |
| Reduced handcrafted | 43 | SVM | 0.3408 |

Results: `outputs/metrics/handcrafted_baselines.csv`.

The reduced representation removes five highly correlated DWT energy features. Its performance varied by classifier, so redundancy removal was evaluated experimentally rather than assumed to improve accuracy.

## 8. Integration with M2 Transformer embeddings

M2 provided **128-dimensional pooled Transformer representations** for all 1,440 audio recordings. `src/validate_m2_features.py` and `src/feature_fusion.py` align recordings by normalized RAVDESS IDs and verify one-to-one matching, actor IDs, emotion labels, split assignments, and finite embedding values.

M1's `sample_id` retains the `03-` audio modality prefix, while M2's pooled export omits it. The merge normalizes this identifier rather than relying on CSV row order. All 1,440 recordings matched; none were unmatched.

The downstream MLP operates on **frozen** representations; the M2 Transformer encoder is not retrained here.

| Experiment | Dimensions | Validation accuracy | Validation Macro-F1 |
|---|---:|---:|---:|
| Transformer only | 128 | 43.75% | 0.3964 |
| **Handcrafted + Transformer** | **176** | **47.92%** | **0.4565** |
| Reduced handcrafted + Transformer | 171 | 47.92% | 0.4514 |

Results: `outputs/metrics/audio_feature_fusion.csv`.

## 9. Final held-out evaluation

The selected handcrafted-only, Transformer-only, and full-fusion MLP configurations were saved and evaluated on **240 recordings from held-out actors 02, 03, 08, and 17**.

| Audio representation | Dimensions | Test accuracy | Test Macro-F1 | Test weighted-F1 |
|---|---:|---:|---:|---:|
| Handcrafted only | 48 | 35.42% | 0.3500 | 0.3512 |
| Transformer only | 128 | 54.58% | 0.5151 | 0.5341 |
| **Handcrafted + Transformer** | **176** | **57.50%** | **0.5445** | **0.5683** |

The fused model improved test Macro-F1 by **0.0293** over Transformer-only and **0.1944** over handcrafted-only. This is an observed result for this fixed actor split, not a statistical-significance claim.

Saved results include `outputs/metrics/final_test_comparison.csv`, per-class classification reports, confusion matrices, sample-level predictions, and plots under `outputs/figures/`.

### Reproducing training and evaluation

With RAVDESS and M2 pooled exports in the expected locations:

```bash
uv run --locked python src/validate_m2_features.py
uv run --locked python src/train_classifier.py
uv run --locked python src/feature_fusion.py
uv run --locked python src/train_final_models.py
uv run --locked python src/evaluate_final_models.py
```

`train_final_models.py` saves three `.joblib` artifacts in `outputs/models/`. Each contains a fitted scaler/classifier pipeline, feature column order, representation name, and label mapping. **Load joblib files only from trusted sources.** A saved fusion classifier does not include M2's raw-audio Transformer encoder.

## 10. Export for M5 audio-visual fusion

`src/export_audio_features.py` exports **unscaled** 176D audio representations, preserving the five metadata fields `sample_id`, `trial_id`, `actor_id`, `emotion_label`, and `split`.

```bash
uv run --locked python src/export_audio_features.py
```

| Output file (under `outputs/features/export/`) | Rows | Columns |
|---|---:|---:|
| `audio_fused_176d.csv` | 1,440 | 181 |
| `audio_fused_176d_train.csv` | 960 | 181 |
| `audio_fused_176d_val.csv` | 240 | 181 |
| `audio_fused_176d_test.csv` | 240 | 181 |

The 176 feature columns are prefixed `m1_` (48 handcrafted) or `m2_` (128 Transformer). The CSVs contain **features, not predictions**.

**For M5:** join audio and visual representations using the modality-independent `trial_id`, then verify actor, emotion, and split consistency. Fit any feature scaling or feature selection on training data only; apply the fitted transformations to validation and test data. The final audio-visual model and its evaluation are separate from the audio-only results reported here.

These exports are intentionally excluded from Git and should be shared separately with the authorized team members.

## 11. Reproducibility and limitations

- Fixed actor-independent partition, with 16 train, four validation, and four test actors.
- Validation Macro-F1 used for model comparison; held-out test used for final evaluation.
- Scaling performed within training-only fitted pipelines.
- MLP internal early stopping used a random subset of the training recordings, **not an actor-group-aware internal split**.
- Results are from one actor partition and seeded training configuration; no multi-split uncertainty or statistical-significance analysis was performed.
- RAVDESS contains acted emotional speech, so performance should not be generalized directly to natural conversational emotion recognition.
- M2 embeddings and encoder configuration are external dependencies. The exported 176D CSV is suitable for dataset-level integration; generating new Transformer embeddings from raw audio requires M2's encoder.
- Raw audio, external embeddings, generated feature matrices, and model binaries are excluded from the Git repository. Code, metadata/splits, evaluation tables, figures, and notebooks can be version-controlled.

## 12. Implementation status

| Component | Status |
|---|---|
| RAVDESS indexing and actor-independent partitioning | Complete |
| Audio preprocessing and full-dataset validation | Complete |
| Time-domain and spectral features (28D) | Complete |
| DWT features and full handcrafted representation (48D) | Complete |
| Handcrafted feature-group classification | Complete |
| M1–M2 recording alignment and audio feature fusion | Complete |
| Final held-out audio evaluation and analysis | Complete |
| 176D feature export for M5 | Complete |
| Group-level audio-visual fusion | Outside M1 scope / M5 integration |

## 13. Data and artifact availability

The original RAVDESS recordings are available from the cited Zenodo dataset. M2 pooled representations and configuration are supplied by the audio deep-learning teammate. Because the repository excludes large/generated files, a fresh clone requires those inputs before the batch feature, fusion, and evaluation scripts can run. The saved model artifacts and feature exports must be transferred separately if another team member needs to use them without retraining.
