# M5_reference_visual_head_sNA_20261019_01 — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

Reference the probability-producing head supplied by the teammate, without claiming a new M5 training run.

Comparison group: `M5_imported_probability_references_v1`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | Imported visual feature export → supplied upstream classifier probabilities (no M5 retraining) | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | probabilities / [None, 8] | Train-only normalization where applicable |
| Fusion | none | False |
| Classifier | upstream classifier (architecture not supplied) | cached_upstream_evaluation |

## 3. Results and selection

Validation Macro-F1=0.3033; test Macro-F1=0.2150. Best epoch=None. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

Upstream training records/checkpoint identity are not fully supplied. Probability class order has been confirmed, but that does not establish identity with the feature extractor checkpoint. The largest test confusion counts include fearful→disgust (22) and angry→disgust (19). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
