# Shared experiment records

See summary.md for pipeline configurations and final results. There are48 M2 records:33 MFCC CNN/BiLSTM/Transformer,5 initial frozen-WavLM experiments and10 downstream continuation experiments. Each schema-v2 run.json separates feature extraction, representation, classifier, training and selection. Every record now has completed validation and test results, per-class metrics, raw count confusion matrix and standardized prediction CSVs. Scores are0–1.

Final test completion was authorized2026-10-26 after model development ended. One prior test record was preserved; the latest selected model's already generated locked delivery probabilities were imported; the remaining46 fixed checkpoints were evaluated. No optimization, augmentation, normalization fitting or model selection occurred. All checkpoint hashes stayed unchanged. Final model remains M2_encoder_wavlm_refinement_s42_20261021_05, selected using validation Macro-F1 before this batch. Do not choose a new winner from the final test ranking.

Historical training summaries, status files, checkpoint flags and protocol snapshots retain their training-time validation-only state; current run.json, test_metrics.json and predictions_test.csv contain final evaluation results. Exact unknown historical times/Git identities remain null. Training-time sources/patches and input manifests are under shared/M2 or the research source snapshots. Zero-context historical patches require git apply --unidiff-zero in a separate checkout.

MFCC checkpoints include the full encoder/head. WavLM checkpoints include downstream adapters/pooling/head only; download the frozen upstream weights using the pinned revision/SHA downloader. Do not replace the legacy MFCC loader with a WavLM checkpoint; use the corresponding research inference loader. Legacy defaults and teammate delivery files are not automatically overwritten. Raw datasets, environments, cached hidden states and local Deliverables are excluded from GitHub.

Final batch protocol, checkpoint lock hashes, all-test-results CSV, report and independent validation are under shared/M2/final_test_20261026/. To verify, from audio/learned run:

```bash
research/speech_transfer/.venv/bin/python scripts/validate_experiment_records.py
research/speech_transfer/.venv/bin/python -m research.speech_transfer.validate_records
research/speech_transfer/.venv/bin/python -m research.speech_transfer.refinement.validate_records
research/speech_transfer/.venv/bin/python scripts/validate_final_tests.py
```

The first three audit each family's validation/test records and checkpoint/provenance compatibility. The final audit checks all11520 test predictions, actors, labels, class order, probabilities, metrics, confusion counts and unchanged locked checkpoint hashes. Older validation reports remain historical evidence, not the current test status.
