# M5_sequence_mean_concat_s42_h64_d0.2_20261019 — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

Compare sequence_mean_concat against fixed-feature controls under the same actor split and training budget.

Comparison group: `M5_sequence_controls_v1`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | sequence_mean_concat on frozen historical audio/visual exports | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | combined / [None, 704] | Train-only normalization where applicable |
| Fusion | sequence_mean_concat | False |
| Classifier | Linear(704,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier |

## 3. Results and selection

Validation Macro-F1=0.6245; test Macro-F1=0.5878. Best epoch=20. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

Test was evaluated only after the candidate/epoch lock; test actors are now exposed for future work. Sequence inputs and projection/normalization differ from pooled baselines; performance changes do not isolate attention alone. The largest test confusion counts include happy→disgust (10) and fearful→sad (10). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
