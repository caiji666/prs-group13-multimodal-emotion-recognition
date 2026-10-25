# M2_encoder_wavlm_refinement_s42_20261021_02

Experiment: freeze_projection. Parent: M2_encoder_wavlm_s42_20261020_05. Configuration and freeze prefixes: training.json. Exact pre-run decision: decision.json. WavLM backbone remains frozen; downstream trainable/frozen counts exclude upstream. New residual layers initialize to identity. Train960/val240 only; test remains not_run/null.

See ../../audio/learned/research/speech_transfer/refinement/README.md for method details. Scores measure the complete pipeline. Exact training times unknown. Frozen normalization is inherited from train-only fitted buffers. Original summary/provenance remain historical evidence; portable parent path is in training.json. No new feature handoff is asserted.
