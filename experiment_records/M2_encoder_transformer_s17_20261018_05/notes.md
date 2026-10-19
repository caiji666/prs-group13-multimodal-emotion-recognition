# M2_encoder_transformer_s17_20261018_05 — Pipeline Experiment Record v2

Owner/member: M2 (personal owner name not recorded). Structured record: [run.json](run.json).

## Question and pipeline

Compare three learned audio pipelines under M2_audio_encoders_20261018_05. MFCC → transformer → masked mean pooling → MLP 128→64→8. All encoder and head parameters are jointly trainable; this is not frozen-encoder classification. Warm-start parent: M2_encoder_transformer_s7_20261018_04. Optimizer/scheduler reset for this run. No multimodal fusion or handcrafted feature combination is performed.

## Result and interpretation

Best validation epoch 2; Accuracy 0.5416666667, Macro-F1 0.5286029632, 240 validation samples. The weakest per-class F1 is disgust (0.3913); strongest is surprise (0.6897). Raw confusion counts and probabilities are retained for interpretation. Scores evaluate the complete encoder/pooling/head pipeline; warm starts, seeds and hyperparameters differ across groups, so cross-group gains cannot be attributed solely to feature quality. These adaptively tuned validation results do not establish test generalization; test has not been run and remains null.

## Selection and handoff

Historical candidate; group-level selection status is recorded separately from current overall default selection. The full checkpoint contains both encoder and MLP. Export sequence before pooling and pooled features before the final MLP using eval/inference mode: variable [Tprime,128], pooled [128], eight learned-only probabilities. Padding mask True means padding; per-sample exported sequences are unpadded. M1 consumes pooled features; M5 consumes sequences and explicitly learned-only pooled/probability outputs. No feature export is claimed merely because this record was created.

## Provenance and limitations

Exact start/finish times remain null. Git base is known; linked patch was reconstructed from verified historical source/configuration, not captured contemporaneously. Hash-matched historical trainer/model/data source snapshots are stored under shared/M2/source_snapshots; some old snapshots were recovered from local historical backups only during migration and are now shipped, with no runtime dependency on those original locations. Original checkpoints/metrics may contain the old surprised label alias; the canonical record uses surprise in the same class position. Checkpoint tensor weights are unchanged. Full sample manifest includes 1440 recordings; ordered training/validation manifest includes 1200; test recordings were indexed, not decoded for training or evaluated.
