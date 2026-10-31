# Group 13 Experiment Summary — v2

Copy this file to `experiment_records/summary.md`. Use the same run IDs in both tables. One configuration row represents one complete pipeline/configuration/seed. One result row represents one evaluated split of that pipeline; validation and test use the same fitted components. Keep the original ID when referencing an upstream baseline.

Copy values from each run's `run.json`; this table is a readable index, not a separate source of truth. Use scores in 0–1 with four decimals. `—` means not evaluated. Template placeholders are not actual results.

## Real RAVDESS speech — pipeline configurations

| Run ID / record link | Member / stage | Comparison group | Feature extractor(s) | Representation / dimensions | Fusion | Classifier / head | Training regime | Seed |
|---|---|---|---|---|---|---|---|---|
| `<run_id>` | `<M1–M5 / stage>` | `<group>` | `<method; upstream run ID>` | `<handcrafted / learned / combined; D>` | `<none / method>` | `<MLP D→H→8 / SVM parameters / none>` | `<frozen / joint / partial fine-tuning / handcrafted / probability fusion>` | `<seed>` |

Feature extractors include temporal encoders such as BiLSTM; the classification head is the downstream mapping to eight classes. A classifier's feature input dimension must match the preceding output. For combined features, show the component dimensions, e.g. `D_handcrafted + D_learned`. For decision fusion, reference the audio/visual probability-producing runs and write `none (uses upstream classifiers)` for the new head.

## Real RAVDESS speech — full-pipeline results

| Run ID | Split | N evaluated | Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Evaluation status |
|---|---|---|---|---|---|---|---|
| `<same run_id>` | val | — | — | — | — | — | not_run |
| `<same run_id>` | test | — | — | — | — | — | not_run |

These are **feature extraction/combination + optional fusion + classifier** results. Do not label them extractor-only scores. Record class-level metrics and confusion matrices in the run record. A `completed` evaluation row requires actual predictions and metrics; the JSON tracks overall run status separately.

## Validation-based selections

| Comparison group | Selected run ID | Selection purpose | Reason / controlled factors / limitations |
|---|---|---|---|
| `<group>` | pending | `<learned encoder / final audio representation / final visual representation / fusion>` | `<validation Macro-F1; head/training protocol held fixed or differences disclosed>` |

Do not compare scores across different sample sets. When comparing extractors, keep the head family, hidden layers, training and search budget as consistent as possible; the first layer may change with input dimension. When comparing heads, keep the exact extracted features fixed. When both change, describe the comparison as complete pipelines.

## Synthetic engineering checks

Use separate copies of the configuration and result tables. Do not use synthetic scores as evidence of RAVDESS performance or to select the final real-data model.

## External sample evaluation

Use separate copies of the tables with dataset version, sample count and labelling protocol. Do not combine with the RAVDESS test set. Keep repeated seeds as individual runs; any mean±std row is an additional explicitly labelled aggregate, never a selected best seed.
