# M2_encoder_wavlm_s42_20261020_03

Frozen upstream WavLM; only adapter/pooling and MLP trained. Training config: training.json. Scores measure full pipeline. Random downstream initialization. See ../../audio/learned/research/speech_transfer/EXPERIMENTS.md for adaptive decisions and comparison limitations. Exact times unknown. Upstream 360MB weights are downloaded by pinned revision/hash; best.pt is downstream only.

Final test evaluation added2026-10-26 after model development ended; fixed checkpoint, no fitting/augmentation/selection. Historical status/summary files retain their training-time validation-only state. Current run.json and test_metrics.json hold final test results.
