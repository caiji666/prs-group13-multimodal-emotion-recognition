# M5_audio_learned_pilot_s42_20261016_01 — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

Validate learned-only audio interface before multimodal inputs were available.

Comparison group: `M5_audio_interface_pilot_v1`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | Frozen M2 Transformer128D → newly trained MLP128→64→8 | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | learned / [None, 128] | Train-only normalization where applicable |
| Fusion | none | False |
| Classifier | Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier |

## 3. Results and selection

Validation Macro-F1=0.4237; test not_run. Best epoch=14. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

20-epoch single-seed pilot with no test evaluation. The original upstream M2 baseline is already recorded as M2_encoder_transformer_s42_20261015_00; do not count it as another training run. The largest validation confusion counts include disgust→angry (13) and neutral→calm (12). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
