# Audio encoder module architecture

## Package layout

```text
audio/learned/
├── ravdess_m2/
│   ├── data.py                 # Speech IDs, shared split, waveform preprocessing, MFCC, batches
│   ├── models.py               # Independent CNN, BiLSTM, Transformer; sequence/pooling/classifier
│   ├── train.py                # Training, validation, early stopping, checkpoints and comparison
│   ├── augmentation_cache.py   # Deterministic train-only MFCC augmentation cache
│   ├── tuning.py               # Replay and summarize the five-round experiment plan
│   ├── dataset_tools.py        # Dataset validation, MFCC extraction and model smoke CLI
│   ├── inference.py            # Frozen encoder WAV prediction and feature extraction
│   ├── feature_io.py           # NumPy readers, ID alignment, padding and bundle validation
│   ├── feature_export.py       # Selected-checkpoint validation and clean dataset feature export
│   ├── fusion_export.py        # M5 learned-only sequences, pooled features and probabilities
│   └── delivery.py             # Portable M1/M5 package, inference source, metadata and ZIP
├── scripts/
│   ├── download_data.py        # Official speech dataset download with checksum
│   └── build_handoff.py        # One-command selected-model export and delivery
├── tests/                     # Data, training, inference, feature export, tuning, integration
├── configs/                   # Preprocessing, training, five tuning rounds and replay plan
├── metadata/                  # Portable shared actor split and canonical label mapping
├── models/                    # Selected tuned weights and selected-run provenance
└── docs/                      # Architecture, feature/delivery guides, experiment reports
```

Small coherent modules remain separate: array readers must work without the training CLI; feature export must not start training; delivery orchestrates already validated outputs; inference packages include only the minimal original inference modules. Combining these would increase dependencies and make standalone handoffs harder to read. Numerical/core modules and selected weights are unchanged.

## Source-name migration

| Previous module | Current module |
|---|---|
| stage4.py | feature_export.py |
| deliver.py | delivery.py |
| m5_preview.py | fusion_export.py |
| experiments.py | tuning.py |
| cli.py | dataset_tools.py |

Imports, tests, README commands and scripts are updated. Use the new module names in new commands. Training and inference entry points remain `ravdess_m2.train` and `ravdess_m2.inference`. Test files use functional names, including `test_audio_data.py` and `test_feature_export.py`. M5 export's internal function is `export_learned_audio`.

## Data flow

WAV → data.py preprocessing/MFCC → models.py encoder → sequence → masked pooling → classifier logits. train.py optimizes each encoder independently and selects using validation Macro-F1. inference.py restores a frozen checkpoint. feature_export.py exports clean representations; fusion_export.py adds learned-only head probabilities; delivery.py assembles existing output contracts.

## Commands (run from audio/learned)

```bash
# Existing dataset validation and MFCC/model diagnostics
python -m ravdess_m2.dataset_tools validate --data /absolute/path/to/speech
python -m ravdess_m2.dataset_tools smoke --data /absolute/path/to/speech
# Three independent candidates, original baseline settings
python -m ravdess_m2.train --encoder all --config configs/train.yaml --data /absolute/path/to/speech --output runs/new_baseline
# Repeat latest warm-start tuning settings using shipped parent weights
python -m ravdess_m2.train --encoder all --config configs/tuning_20261017/best_resolved.yaml --data /absolute/path/to/speech --output runs/new_refinement
# Single candidate: replace all with cnn, bilstm, or transformer
python -m ravdess_m2.tuning --plan configs/tuning_20261017/plan.json --output runs/new_tuning
python -m ravdess_m2.feature_export export --run models/selected_run --data /absolute/path/to/speech --output artifacts/new_features
python -m ravdess_m2.delivery --features artifacts/new_features --output deliveries/new_handoff --zip
python scripts/build_handoff.py --data /absolute/path/to/speech
```

A new clone includes selected, baseline and exact refinement-parent weights under models/. Neither refinement nor five-round replay requires ignored historical runs. Fresh training uses configs/train.yaml. Repeating refinement is not fresh-from-scratch training and is not guaranteed to improve the saved model.

## Unchanged output contract

Existing directories and generated filenames remain identical, including `stage5_contract.json`, `provenance/stage3_*`, `handoff_reader.py`, `features/pooled/`, `m5_learned_feature_only/`, sequence lengths/masks, probability keys, checkpoint schema and inference package filenames. Historical stage names in serialized output/provenance are intentionally retained. Renaming source modules does not rename experiment IDs or artifacts.

## Refactor verification

- 27 existing/updated tests passed.
- data.py, models.py, train.py, inference.py, feature_io.py and augmentation_cache.py are byte-for-byte unchanged.
- All checkpoint files retain their pre-refactor SHA-256.
- Exported one real speech recording from each split using both pre-refactor and refactored code, then built both portable M1/M5 packages and ZIPs. All 73 non-ZIP files have identical relative paths and SHA-256, including sequences, pooled features, probabilities, metadata, manifests and packaged inference source. ZIP containers may have different filesystem timestamps; their content is equivalent.
- No formal retraining, final test-set scoring, full-dataset regeneration, commit or push was performed.
