# WavLM continuation results — 2026-10-21

Baseline: validation Accuracy 0.812500; Macro-F1 0.806292.

All results below are validation-only; no test audio or test predictions/metrics were used in this series. WavLM backbone remains frozen. Parent 0 means original WavLM Round05. Each row uses its own maximum-validation-F1 checkpoint.

| Round | Experiment | Parent | Accuracy | Macro-F1 | Best epoch / completed |
|---|---|---:|---:|---:|---:|
| 01 | head_only | 0 | 0.808333 | 0.802598 | 5 / 19 |
| 02 | freeze_projection | 0 | 0.800000 | 0.791611 | 13 / 27 |
| 03 | gentle_full_adapter | 0 | 0.820833 | 0.810180 | 18 / 32 |
| 04 | residual_frame | 3 | 0.820833 | 0.809716 | 1 / 15 |
| 05 | temporal_context | 3 | 0.841667 | 0.831218 | 2 / 16 |
| 06 | deeper_head | 5 | 0.820833 | 0.811222 | 1 / 15 |
| 07 | freeze_head_refine_sequence | 5 | 0.816667 | 0.804486 | 2 / 16 |
| 08 | feature_noise_mixup | 5 | 0.808333 | 0.799298 | 13 / 27 |
| 09 | focal_classification | 5 | 0.808333 | 0.794674 | 6 / 20 |
| 10 | final_conservative | 5 | 0.816667 | 0.806847 | 2 / 16 |

Winner: round 05; checkpoint `research/speech_transfer/runs/refinement_20261021/round_05_temporal_context/best.pt`.
Accuracy gain: 2.9167 percentage points. Macro-F1 gain: 0.024926.

See ../../refinement/README.md for every planned change, inherited settings and reproduction commands. decision_NN.json stores effective pre-run configurations and parent hashes. The new sequence projection and kernel3 depthwise residual temporal convolution precede attentive-statistics pooling. Sequence and pooled dimensions are128.

The winner has 202/240 correct validation predictions. Baseline had195/240. This small actor validation set has been reused for tuning, so the improvement is development-set evidence, not an independent generalization estimate. Earlier disclosed test results cannot be unseen; no fresh test claim is made and no previous test score is used as a tuning objective.

No model promotion, teammate feature re-export, Git commit or push occurred for this continuation series. Existing model weights and delivery files remain unchanged.
