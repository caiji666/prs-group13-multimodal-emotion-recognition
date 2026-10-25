# M2_encoder_wavlm_refinement_s42_20261021_05

Experiment: temporal_context. Parent: M2_encoder_wavlm_refinement_s42_20261021_03. Configuration and freeze prefixes: training.json. Exact pre-run decision: decision.json. WavLM backbone remains frozen; downstream trainable/frozen counts exclude upstream. New residual layers initialize to identity. Train960/val240 only; test remains not_run/null.

See ../../audio/learned/research/speech_transfer/refinement/README.md for method details. Scores measure the complete pipeline. Exact training times unknown. Frozen normalization is inherited from train-only fitted buffers. Original summary/provenance remain historical evidence; portable parent path is in training.json. No new feature handoff is asserted.

Final test evaluation added2026-10-26 after model development ended; fixed checkpoint, no fitting/augmentation/selection. Historical status/summary files retain their training-time validation-only state. Current run.json and test_metrics.json hold final test results.
