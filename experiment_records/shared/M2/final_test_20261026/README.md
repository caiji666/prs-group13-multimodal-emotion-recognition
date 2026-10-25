# Final fixed-checkpoint test completion —2026-10-26

All48 M2 experiment records are test-completed.46 checkpoints received new inference; one original WavLM test record was preserved and the final convolutional model reused its already validated locked delivery probabilities. All use the shared240-record test split, no augmentation or fitting. Each checkpoint also passed8 saved validation-probability compatibility checks before new evaluation. No raw feature tensors or datasets are published.

all_test_results.csv summarizes accuracy, macro-F1, macro-precision and macro-recall; per-run directories contain standardized predictions, class metrics, confusion counts and current run.json. protocol.json locks all weight hashes. evaluation_report.json records outcomes; validation_report.json independently checks all11520 test predictions. final_model_selection.json identifies the previously validation-selected final model; test ranking does not reselect it.

Original historical summaries/status/checkpoint flags remain immutable training-time evidence. Current final results live in run.json/test_metrics.json. The shared test set was previously used to report an earlier model, so this batch is a final retrospective comparison, not a newly collected or previously unseen dataset. No new development follows these results.

Reproduce inference only with the provided cached/data prerequisites from audio/learned: `research/speech_transfer/.venv/bin/python scripts/evaluate_all_records.py --output PATH_TO_NEW_REPORT_FOLDER`. Completed canonical records are preserved rather than overwritten. Run scripts/validate_final_tests.py to audit current records.
