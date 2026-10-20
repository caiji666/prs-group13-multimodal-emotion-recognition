# M2 — Learned Audio Emotion Recognition

Independent CNN, BiLSTM and Transformer encoders for the RAVDESS speech subset. Includes audio preprocessing/MFCC sequences, training, validation-only model selection, WAV inference, feature export and portable M1/M5 deliveries. Audio only; no video processing.

The supplied 2026-10-18 round04 Transformer has **validation Accuracy 54.58%, Macro-F1 0.532837**. It wins both metrics among saved validation-selected checkpoints. This is a warm-start refinement result; no final test score is claimed.

## 1. Clone and initialize (macOS / Linux)

Requirements: Git, Python **3.9–3.12** with venv, network access for packages and Zenodo, and permission to read this repository. Use Python 3.9 with requirements-tested.txt to match the recorded numerical environment; other supported Python versions use requirements.txt and may give different numerical results. If GitHub requests authentication, sign in with your own authorized account. Ensure python3 points to the intended version.

```bash
git clone https://github.com/caiji666/prs-group13-multimodal-emotion-recognition.git
cd prs-group13-multimodal-emotion-recognition/audio/learned
python3 --version
bash scripts/setup_project.sh --download-data
```

**All following commands run from audio/learned.** The setup script creates an independent .venv, installs dependencies, downloads the official `Audio_Speech_Actors_01-24.zip`, verifies MD5 `bc696df654c87fed845eb13823edef8a`, extracts it to data/raw, validates the complete speech subset and runs tests. It rejects an incomplete existing data/raw. Nothing is read from another local project; activation is unnecessary when using .venv/bin/python.

Expected data layout:

```text
data/raw/Actor_01/03-01-01-01-01-01-01.wav
...
data/raw/Actor_24/*.wav
```

Expected totals: **1440 WAVs; train 960, validation 240, test 240**. Official source: https://zenodo.org/records/1188976 . Existing complete data/raw is validated and preserved. For a failed download, rerun setup; partial downloads are replaced. For checksum mismatch, remove the downloaded ZIP as instructed before retrying. Never delete a valid dataset to resolve a setup error. Setup without --download-data installs dependencies and tests only.

## 2. Validate data and model shapes

```bash
.venv/bin/python -m ravdess_m2.dataset_tools validate --data data/raw --output artifacts/data_validation
.venv/bin/python -m ravdess_m2.dataset_tools smoke --data data/raw --output artifacts/model_shapes
```

Validation decodes the audio and checks MFCCs; smoke checks all three model output shapes with randomly initialized weights. These are engineering checks, not emotion recognition benchmarks.

## 3. Predict a recording using the supplied best model

This example uses a downloaded RAVDESS recording and runs without training:

```bash
.venv/bin/python -m ravdess_m2.inference --checkpoint models/best_audio_encoder.pt --audio data/raw/Actor_01/03-01-01-01-01-01-01.wav --splits ../../multimodal_fusion/configs/actor_splits.csv --output artifacts/example_prediction.npz
.venv/bin/python -c 'import numpy as np; d=np.load("artifacts/example_prediction.npz",allow_pickle=False); print("Emotion:",d["class_names"][d["probabilities"].argmax()]); print(dict(zip(d["class_names"].tolist(),d["probabilities"].tolist())))'
```

For your own WAV, replace --audio with its path and replace --splits with --external. External examples have unknown ground-truth label (-1). Outputs include eight probabilities, logits, sequence, pooled feature, valid length/mask and metadata. These probabilities come from M2's learned-only head. Existing output files are rejected; choose a new filename when repeating.

## 4. Train three independent models

Repeat the current selected configuration using its archived per-encoder parent weights:

```bash
.venv/bin/python -u -m ravdess_m2.train --encoder all --data data/raw --config configs/tuning_20261018/best_resolved.yaml --output "runs/refinement_$(date +%Y%m%d_%H%M%S)"
```

Use --encoder cnn, bilstm or transformer for one independent model. This does not overwrite the supplied best model. A new run is not guaranteed to improve it. For fresh training from random initialization:

```bash
.venv/bin/python -u -m ravdess_m2.train --encoder all --data data/raw --config configs/train.yaml --output "runs/baseline_$(date +%Y%m%d_%H%M%S)"
```

For a bounded two-epoch diagnostic, add --smoke and use a new output directory; diagnostic checkpoints cannot be exported as formal learned features. Training records best.pt, validation predictions/metrics, histories, protocol and comparison files. Model selection uses validation Macro-F1; test audio is excluded from training and validation.

Replay the five-round tuning plan (a longer optional workflow):

```bash
.venv/bin/python -u -m ravdess_m2.tuning --plan configs/tuning_20261017/plan.json --output "runs/tuning_$(date +%Y%m%d_%H%M%S)"
```

The replay plan uses included models/baseline; the latest refinement uses models/refinement_parents. Neither requires ignored historical runs. Different software/hardware may change exact scores.

## 5. Generate M1/M5 deliveries without training

Use the shipped selected model:

```bash
.venv/bin/python scripts/build_handoff.py --data data/raw
```

The script prints the new timestamped artifact/delivery paths. Under deliveries/tuned_TIMESTAMP:

- **M1:** features/pooled/{train,val,test}.csv or NPZ, plus audio_config.json at delivery root. CSV has f_000–f_127 learned feature columns.
- **M5:** m5_learned_feature_only/, with per-sample F_sequence[T,128], F_audio_learned_feature_only[128], P_audio_learned_feature_only[8], sample IDs, split, labels, actor IDs, lengths/masks; split-level pooled/probability files are also included. M5 handles sequence batching/padding.
- **Portable inference:** best_audio_encoder.pt, audio_inference.py, minimal ravdess_m2 source, configuration and requirements. A full delivery ZIP is generated too.

For a **newly trained formal three-model run**, replace runs/YOUR_RUN with the actual completed run directory:

```bash
.venv/bin/python -m ravdess_m2.feature_export export --run runs/YOUR_RUN --data data/raw --output artifacts/YOUR_RUN_features
.venv/bin/python -m ravdess_m2.delivery --features artifacts/YOUR_RUN_features --output deliveries/YOUR_RUN --zip
```

Use fresh output directories each time. Feature export uses eval/inference mode, no augmentation, no fitting and no test metrics. The current pipeline requires a formal comparison of all three candidates before selected-feature export. A single-candidate training run alone is not a completed model comparison.

For M5's audio_pilot ZIP importer, ZIP the m5_learned_feature_only folder with that folder directly at ZIP root. The full delivery ZIP has an additional outer directory and is not that importer's direct input. Formal M1 handcrafted+learned F_audio and final classifier P_audio are M1 outputs; M2 learned-only outputs are the temporary fusion-testing interface.

## Shared contracts

Use ../../multimodal_fusion/configs/actor_splits.csv, matching actors_m2.json; never randomly repartition. metadata/actor_splits.csv is a byte-identical portable copy. Six-field IDs retain leading zeros: 03-01-06-01-02-01-12.wav → 01-06-01-02-01-12. Join by ID, never row order; verify labels, actors and splits.

Labels 0–7: neutral, calm, happy, sad, angry, fearful, disgust, **surprise**. M1 adapts its seven-field IDs to our six-field IDs when joining. Per-sample sequences remain unpadded; True in padding_mask means padding. No one-to-one audio/video frame alignment is assumed.

Only the original class-name metadata changed from surprised to surprise; all model tensors were preserved. Historical protocol hashes remain historical. Feature export permits exactly the documented reversible label-literal migration in data.py; other model/data source changes still fail provenance checks. See docs/checkpoint_migration.json.

## Source and Git scope

- ravdess_m2/: data.py, models.py, train.py, tuning.py, dataset_tools.py, inference.py, feature_io.py, feature_export.py, fusion_export.py, delivery.py, augmentation_cache.py.
- models/: selected weights/provenance, baseline and exact refinement parents, all explicitly allowed in Git.
- configs/, metadata/, scripts/, tests/, docs/: configuration, shared split copy, setup/download/handoff scripts, tests and guides.
- .venv/, data/, runs/, artifacts/, deliveries/: local/generated, ignored by Git. Setup reconstructs environment/data on each machine.

See [architecture](docs/CODE_ARCHITECTURE.md), [feature handoff](docs/FEATURE_HANDOFF.md), [delivery guide](docs/DELIVERY_GUIDE.md), [all five experimental results](docs/experiment_results.csv), and [repository-only workflow](docs/SELF_CONTAINED_WORKFLOW.md). Historical workflow notes describe previous runs; this README is the current entry guide.

## Local follow-up experiments — 2026-10-18

Five additional full three-encoder experiments are archived in the unified records. Best: round04 Transformer, validation Accuracy 54.58%, Macro-F1 0.532837. Accuracy did not improve over the shipped baseline; F1 gain is only 0.000434. The previous default has since been replaced by the round04 model; historical records preserve the previous weights. See [results and per-round changes](docs/TUNING_20261018.md). All 33 historical model checkpoints and records are included in this repository.

## Unified experiment records and current selection — 2026-10-20

All 33 real M2 trained pipelines and full checkpoints are published under [experiment_records](../../experiment_records/summary.md), following schema v2. Current default: M2_encoder_transformer_s7_20261018_04, validation Accuracy0.5458333333, Macro-F10.5328369004; no test result. models/best_audio_encoder.pt, models/selected_run and build_handoff all refer to this checkpoint. Older models remain in their own records. Existing feature handoffs are not automatically regenerated. Warm-start configs now reference published experiment record directories, so old ignored runs are not required for replay. Historical source and reconstructed patches are archived alongside records.

Validate all uploaded M2 records and model aliases:

```bash
.venv/bin/python scripts/validate_experiment_records.py
```

## WavLM pretrained speech extension

Five new isolated experiments and inference/export code are in [research/speech_transfer](research/speech_transfer/README.md). Best Round05: validation Accuracy 81.25%, Macro-F1 0.806292; final locked test Accuracy 82.9167%, Macro-F1 0.828012. WavLM remains frozen; adapters/pooling/head train. This uses external pretraining and raw waveforms rather than MFCC. The original three encoders and legacy default are preserved.

Published best downstream weights: `../../experiment_records/M2_encoder_wavlm_s42_20261020_05/best.pt`. Download the separate pinned WavLM backbone using the extension downloader before inference. Shared-format records cover all five rounds in root experiment_records. Local-only M1/M5 handoff (excluded from GitHub): root `Deliverables/wavlm_round05/m5_learned_feature_only/` (128-dimensional learned pooled vectors, sequences, probabilities). Final test metrics are adjacent in `final_test_evaluation/`.
