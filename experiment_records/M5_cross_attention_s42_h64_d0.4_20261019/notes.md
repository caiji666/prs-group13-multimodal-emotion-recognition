# M5_cross_attention_s42_h64_d0.4_20261019 — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

Compare cross_attention against fixed-feature controls under the same actor split and training budget.

Comparison group: `M5_regularization_exploration_v1`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | cross_attention on frozen historical audio/visual exports | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | combined / [None, 64] | Train-only normalization where applicable |
| Fusion | cross_attention | True |
| Classifier | Linear(64,64) → ReLU → Dropout(0.4) → Linear(64,8) | frozen_encoder_then_classifier |

## 3. Results and selection

Validation Macro-F1=0.6828; test not_run. Best epoch=15. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

Only validation was evaluated for this exploratory regularization trial. Sequence inputs and projection/normalization differ from pooled baselines; performance changes do not isolate attention alone. The largest validation confusion counts include surprise→sad (12) and sad→disgust (7). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
