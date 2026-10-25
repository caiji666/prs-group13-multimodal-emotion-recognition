# Shared experiment records

Start with summary.md. The 33 M2 directories follow docs/templates/run.template.json v2. Each directory contains run.json, notes.md, best.pt (full encoder+head), validation predictions, metrics/confusion counts, history, status and a portable training YAML. All model weights are versioned explicitly; raw datasets/features/environments are not included. Artifact references in run.json are repository-root relative.

Historical run/source provenance and full 1440-sample manifest are under shared/M2. Exact unknown times and pre-integration Git identities are null. Source snapshots are historical evidence, not a standalone package; current audio/learned is the maintained runtime. For 20261018, code patches are reconstructed against the known Git base and labeled accordingly. Warm-start checkpoints whose metadata was canonicalized have both the original input SHA and published SHA recorded. Model tensors and prediction values are unchanged.

Validation scores are full-pipeline scores, not isolated feature-quality measurements. Test remains not_run/null for all records. Current default is M2_encoder_transformer_s7_20261018_04; no new full-dataset feature handoff is asserted by recording experiments.

Historical patches use zero context to avoid treating patch context spaces as source whitespace. For forensic replay against their declared base, use git apply --unidiff-zero in a separate checkout; current runtime already contains later changes.

Validation: shared/M2/validation_report.json and integration_validation.json. The exact Git-index export passed schema/metric/checkpoint checks and 29 unit tests; default full-data export matched all240 selected validation predictions. Generated feature tensors and diagnostic smoke outputs remain outside Git. Run `cd audio/learned && .venv/bin/python scripts/validate_experiment_records.py` to recheck the published records.

## WavLM extension

Five additional schema-v2 WavLM records bring M2 total to 38. Their upstream encoder is frozen, with downstream adapters/pooling/head trained. Only selected Round05 has final test metrics; prior 33 and WavLM rounds01–04 remain not_run/null. See summary.md. WavLM best.pt files require separately downloaded pinned upstream weights. New records are checked by `python -m research.speech_transfer.validate_records`; the historical validator covers only the original 33 MFCC records.

## Ten WavLM continuation records

`M2_encoder_wavlm_refinement_s42_20261021_01` through `_10` add ten validation-only records, bringing M2 total to48. Every record preserves weights, config, history, validation probabilities/metrics, parent and source hashes. All ten test states are not_run/null. Validate with `python -m research.speech_transfer.refinement.validate_records` from audio/learned. Current research validation winner is continuation05 (Accuracy0.841667, Macro-F10.831218); legacy defaults and existing handoffs remain unchanged.
