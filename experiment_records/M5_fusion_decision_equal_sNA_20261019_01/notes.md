# M5_fusion_decision_equal_sNA_20261019_01 — Pipeline Experiment Record v2

Owner/member: M5 Jiang Yushan  
Structured record: [run.json](run.json)

## 1. Question and comparison

Compare fixed equal weighting and validation-tuned probability weighting, holding source probability exports fixed.

Comparison group: `M5_decision_fixed_exports_v1`. Metrics evaluate the complete recorded pipeline, not feature extraction alone.

## 2. Pipeline at a glance

| Component | Method / structure | Training relationship |
|---|---|---|
| Feature extraction | decision_equal using supplied audio/visual heads | Frozen supplied features/probabilities; upstream checkpoints recorded only where verified |
| Representation | probabilities / [None, 8] | Train-only normalization where applicable |
| Fusion | decision_equal | False |
| Classifier | none | probability_fusion |

## 3. Results and selection

Validation Macro-F1=0.4906; test Macro-F1=0.5579. Best epoch=None. Epochs and weights were selected using validation only. Selection is at the comparison-group/method level; no best-test seed was chosen.

## 4. Interpretation and limitations

Class order was confirmed after the initial evaluation and values remained unchanged. Weight selection used validation only; test performance did not change alpha. No new classifier was trained. The largest test confusion counts include sad→calm (9) and angry→disgust (8). These are descriptive findings, not proof of a causal feature explanation. Unknown upstream metadata remain null rather than being inferred from a model name.

## 5. Handoff and next step

Prediction CSVs use the shared v2 schema and fixed class order. Source code, input manifest, diagnostic plots, locks and hashes are referenced in run.json and `experiment_records/shared/M5/`. Large features and M5 checkpoint weights are not uploaded; their hashes/availability are listed in the shared artifact inventory. This record describes the historical feature version supplied before 2026-10-20, not M2's newly promoted round04 export. See the shared M5 report for class-order confirmation and test-exposure chronology.
