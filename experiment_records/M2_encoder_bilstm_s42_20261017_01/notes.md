# M2_encoder_bilstm_s42_20261017_01 — Pipeline Experiment Record v2

Owner/member: M2 (personal owner name not recorded). Structured record: [run.json](run.json).

## Question and pipeline

Compare three learned audio pipelines under M2_audio_encoders_20261017_01. MFCC → bilstm → masked mean pooling → MLP 128→64→8. All encoder and head parameters are jointly trainable; this is not frozen-encoder classification. Random initialization; no warm-start parent. No multimodal fusion or handcrafted feature combination is performed.

## Result and interpretation

Best validation epoch 15; Accuracy 0.3875000000, Macro-F1 0.3170831891, 240 validation samples. The weakest per-class F1 is neutral (0.0000); strongest is calm (0.5357). Raw confusion counts and probabilities are retained for interpretation. Scores evaluate the complete encoder/pooling/head pipeline; warm starts, seeds and hyperparameters differ across groups, so cross-group gains cannot be attributed solely to feature quality. These adaptively tuned validation results do not establish test generalization; test has not been run and remains null.

## Selection and handoff

Historical candidate; group-level selection status is recorded separately from current overall default selection. The full checkpoint contains both encoder and MLP. Export sequence before pooling and pooled features before the final MLP using eval/inference mode: variable [Tprime,128], pooled [128], eight learned-only probabilities. Padding mask True means padding; per-sample exported sequences are unpadded. M1 consumes pooled features; M5 consumes sequences and explicitly learned-only pooled/probability outputs. No feature export is claimed merely because this record was created.

## Provenance and limitations

Exact start/finish times remain null. Original Git commit/dirty state and exact historical command were not captured; these remain null rather than substituting the current commit. Hash-matched historical trainer/model/data source snapshots are stored under shared/M2/source_snapshots; some old snapshots were recovered from local historical backups only during migration and are now shipped, with no runtime dependency on those original locations. Original checkpoints/metrics may contain the old surprised label alias; the canonical record uses surprise in the same class position. Checkpoint tensor weights are unchanged. Full sample manifest includes 1440 recordings; ordered training/validation manifest includes 1200; test recordings were indexed, not decoded for training or evaluated.
