# Group 13 — Multimodal Emotion Recognition

PRS Practice Module: **Multimodal Emotion Recognition Using Speech and Facial Expressions**.
This private repository is the team's shared code workspace for the RAVDESS speech subset (24 actors, eight emotions).

## Project layout

| Directory | Purpose | Current state |
|---|---|---|
| `audio/` | M1 handcrafted and M2 learned audio encoders | Implemented under `handcrafted/` and `learned/` |
| `visual/` | Face preprocessing, handcrafted features and learned encoders | Reserved for teammates' code |
| `multimodal_fusion/` | Decision, concatenation, gated and cross-attention fusion | Existing implementation |
| `demo/` | External audio-video prediction interface | Reserved; not implemented |
| `docs/` | Submission checklist, integration and collaboration notes | Initial guidance |

The current upload contains the fusion implementation, tests and shared actor split. The full end-to-end system is not complete. Synthetic experiments are engineering checks, not RAVDESS results. A real M2 learned-only audio pilot has been performed locally; real visual fusion results are still pending. Generated results, datasets and checkpoints are not included in this initial upload.

## Run the fusion engineering check

```bash
git clone https://github.com/caiji666/prs-group13-multimodal-emotion-recognition.git
cd prs-group13-multimodal-emotion-recognition/multimodal_fusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m fusion demo --output results/my_first_demo --epochs 30
```

See [fusion documentation](multimodal_fusion/README.md) for feature export and training. Download and prepare RAVDESS separately; the code-only checkout does not contain data or trained weights.

## Shared integration contract

- Use the agreed [actor split](multimodal_fusion/configs/actor_splits.csv): 16 train / 4 validation / 4 test actors. Do not randomly repartition samples.
- For **every real-data fusion command**, explicitly use `--split configs/actors_m2.json` from inside `multimodal_fusion/`. The CLI now defaults to the formal `actors_m2.json`; `actors.json` retains the historical development split.
- Derive `sample_id` by removing only the modality field and extension: `03-01-06-01-02-01-12.wav` → `01-06-01-02-01-12`. Preserve leading zeros, and join modalities by ID.
- Preserve labels, actor/split metadata and probability order: `neutral, calm, happy, sad, angry, fearful, disgust, surprise` (labels 0–7).
- Select models and fusion weights on validation macro-F1. Reserve the test actors for final evaluation.
- Compare audio-only, visual-only and all four fusion methods. Cross-attention is included in the team's later agreed experiment scope.

Detailed interfaces and experiment plans are in [the integration document](multimodal_fusion/实验规划与当前代码说明.md).

## Team workflow

Each member works on a module branch, such as `feature/audio-handcrafted`, `feature/audio-learned`, `feature/visual`, `feature/fusion` or `feature/evaluation-demo`, and opens a pull request into `main`. These are suggested branch names, not a reassignment of the team's existing responsibilities.

Before merging, include the entry command, dependencies, input/output fields, validation results and any interface changes. Keep machine-specific dataset paths in ignored local config. Store large datasets and trained weights separately with retrieval instructions and checksums; they still need to be accounted for in the final course submission.

Team responsibilities (confirmed by the shared allocation table): M1 owns handcrafted audio and the audio representation comparison; M2 owns learned audio encoder comparison; M3 owns handcrafted visual features and the visual representation comparison; M4 owns learned visual encoder comparison; M5 owns multimodal fusion and unimodal/multimodal comparison. Jiang Yushan owns M5. M1 combines M1/M2 outputs; M3 combines M3/M4 outputs; M5 consumes the selected audio and visual outputs.

Use the [shared experiment recording format](docs/experiment-recording.md) and [templates](docs/templates/summary.template.md) to record comparable configurations, upstream runs, validation selection and final results.

## Course submission

GitHub is a collaboration workspace. The course handout does not require a GitHub repository or a public repository. Final submission is **one team ZIP on Canvas by 31 October 2026**, with individual reports submitted separately. See [the requirements checklist](docs/course-requirements.md).

## M1 Handcrafted Audio Module

See [audio/handcrafted/README.md](audio/handcrafted/README.md) for the complete handcrafted audio pipeline, preprocessing experiments, classifier comparisons, hyperparameter tuning, audio feature fusion, and M5 handover.

### Implementation and Results

- **Dataset:** RAVDESS speech subset — 1,440 recordings, 24 actors, eight emotions.
- **Handcrafted features:** 48D representation combining time-domain (12D), spectral (16D), and wavelet (20D) features.
- **Preprocessing experiments:** 30 configurations using five wavelets, two normalization settings, and three frame lengths.
- **Classifier comparison:** Nine classifiers evaluated across 270 configurations, followed by hyperparameter tuning of five shortlisted models.
- **Selected M1 classifier:** Logistic Regression (`C=1.0`) with db2 wavelet, normalized audio, and 20 ms frames.
- **M1 validation performance:** Accuracy **40.00%**, Macro-F1 **0.3885**.

### M1–M2 Audio Feature Comparison

M1 also evaluated the integration of handcrafted features with M2's exported deep-learned audio representation (final WavLM Round 05 export, 128D pooled features).

| Feature Representation | Dimensions | Validation Macro-F1 |
|---|---:|---:|
| Handcrafted only | 48D | 0.3885 |
| Deep-learned only (WavLM) | 128D | **0.7999** |
| Combined features | 176D | 0.7891 |

Results use the same StandardScaler + Logistic Regression pipeline (`C=1.0`). The WavLM features outperformed the handcrafted features for every emotion, and simple feature concatenation did not outperform the deep-learned representation alone. An earlier run against M2's MFCC Transformer export (deep-only 0.4734, combined 0.4518) is archived; see [the M1 comparison document](audio/handcrafted/docs/04_audio_feature_fusion_comparison.md).

### M5 Handover

M1 has prepared the following outputs for multimodal fusion:

- **48D handcrafted features** for training, validation, and testing.
- **Eight-class probabilities**, including actor-independent out-of-fold predictions for training.
- **Fitted Logistic Regression model** and feature/class metadata.
- **Aligned recording identifiers** and preserved actor-independent partitions.
- **Audio representation selection:** WavLM learned-only features scored highest, so the audio input for fusion is M2's WavLM delivery, loaded directly by M5. Optional ready-to-load NPZ files are also available: `audio_wavlm.npz` (M2's WavLM data as one file, validation Macro-F1 0.8063) and `audio_combined.npz` (handcrafted + WavLM 176D variant, 0.7891).

See [M1 handover documentation](audio/handcrafted/docs/05_m1_audio_handover.md) for file schemas, probability ordering, and integration instructions.

The M1–M2 feature comparison and M5 handover are complete.

## M2 learned audio module

See [audio/learned/README.md](audio/learned/README.md) for full Stage 1–6 source, selected tuned Transformer weights, WAV emotion inference, training and M1/M5 exports. The shipped model has validation Accuracy 54.58% and Macro-F1 0.532837 (2026-10-18 round 04); final test generalization remains unmeasured. Six-field IDs and per-sample sequences are unchanged; class 7 is named `surprise`.

M2 has published [33 canonical experiment records and checkpoints](experiment_records/summary.md), including predictions, history and source provenance. All M2 test results remain not_run/null.

## M5 published results (2026-10-20)

M5 has published 30 schema-v2 records using the team's [recording format](docs/experiment-recording.md). See the [shared experiment summary](experiment_records/summary.md) and [M5 results and remaining work](experiment_records/shared/M5/M5_RESULTS_20261020.md). Records include configurations, validation/test predictions, histories, source snapshots and interpretation figures. Large M5 weights and raw features are not uploaded; hashes and retrieval status are explicit.

These runs use the historical M2 feature export associated with the initial Transformer checkpoint, not the newly promoted round04 default. Cross-attention was selected on validation; gated had a higher observed test mean, without test-based reselection. Visual probability class order is confirmed.

Validate the published M5 records without private feature data or checkpoints:

```bash
python scripts/validate_m5_records.py --root .
```
