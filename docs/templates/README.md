# Shared experiment templates — quick start

These templates support **all five members (M1–M5)**. They describe the actual pipeline; they do not require everyone to use an MLP or train the encoder and classifier separately.

## What to fill

1. Create `experiment_records/<unique_run_id>/` at the repository root.
2. Copy [run.template.json](run.template.json) there as `run.json`, and [experiment.template.md](experiment.template.md) as `notes.md`.
3. Fill the feature extractor(s), representation, optional fusion, classifier and training regime. Use `null` for unknown/not-applicable scalar fields and explain them in notes. Do not invent values.
4. Record validation metrics. Leave test as `null` / `not_run` until final evaluation of the locked configuration. Scores use 0–1.
5. Add the pipeline and result rows to `experiment_records/summary.md`, using [summary.template.md](summary.template.md). Use the same run ID in both tables.

One run represents one full pipeline/configuration/seed. Classification scores evaluate the full pipeline, not the extracted features in isolation. Keep per-class metrics, confusion counts and per-sample predictions for evaluated splits. Existing program outputs may be linked instead of rewritten; the templates do not automatically convert or validate those outputs.

## How each member uses the same fields

| Member | Feature extractor / representation | Classifier | Special instructions |
|---|---|---|---|
| M1 — Audio handcrafted | Handcrafted audio; for combined runs also reference M2's selected learned extractor | Actual handcrafted classifier or combined-feature MLP | Add both extractor entries for combined features and record component dimensions |
| M2 — Audio learned | MFCC → CNN / LSTM-BiLSTM / Transformer → pooling | MLP with explicit layers, activation and dropout | Record frozen, joint or partial fine-tuning; select the learned encoder using validation Macro-F1 |
| M3 — Visual handcrafted | DWT/texture features + temporal pooling; for combined runs also reference M4 | SVM for DWT baseline; actual MLP for combined representation | Put SVM kernel/C/gamma/calibration in classifier parameters; MLP-only fields stay null |
| M4 — Visual learned | MobileNetV3-Small or ResNet18; include BiLSTM in the extractor where used | MLP with explicit structure | Record frame/face preprocessing, pooling, pretraining and fine-tuned layers |
| M5 — Multimodal | Selected audio/visual inputs, with source run IDs | Actual downstream head, or none for decision fusion | Record fusion method and parameters separately from the classifier |

For single-modality runs, set `pipeline.fusion.applied=false`; unused fusion fields remain null/empty. For decision fusion, use `feature_extractors=[]` if only cached probabilities are consumed, set representation kind to `probabilities`, and reference the actual audio/visual runs in `fusion.input_source_runs`. Set classifier type to `none`, trainable to false and dimensions/MLP fields to null; the output still contains eight probabilities from fusion. For feature fusion, keep the extractors' provenance even when they are frozen.

Keep `source_runs` entries that do not apply as null. For an upstream learned-only baseline already evaluated by M2/M4, link its original run rather than claiming a new independent training run. A new head trained on the same cached features is a new run that references the source.

## Fair comparisons

- Use the shared actor split, class order and sample IDs. Compare the same evaluation samples.
- Distinguish learned encoder selection (M2/M4), final representation selection (M1/M3), and fusion selection (M5).
- When comparing extractors, keep the head family and training protocol consistent where possible; disclose changes in input dimension and parameter count.
- Do not describe a DWT-SVM versus learned-MLP comparison as isolating feature quality: both representation and classifier differ.
- Keep synthetic checks and external evaluations separate from real RAVDESS results. Unmeasured scores stay null, never zero.

Detailed protocol and existing M5 output mapping: [experiment recording guide](../experiment-recording.md).
