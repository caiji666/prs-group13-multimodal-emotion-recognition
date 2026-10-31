# Current implementation status

Sources: supplied Group 13 proposal, work division, original actor_splits.csv and Sample ID convention. References and the original split CSV are preserved.

Stages 1/2 are implemented: speech ingestion, strict actor split, shared IDs, waveform preprocessing, train-only augmentation, MFCC [T,40], independent CNN/BiLSTM/Transformer, masked global pooling and 8-class logits.

Stage 3 is implemented and was run manually by the user: run stage3_20261015_194625 completed. Transformer epoch 14 was selected by validation macro-F1 0.45082615. Training-only MFCC standardization is checkpointed. Test scores have not been computed.

Stage 4 is completed: selected Transformer checkpoint used in frozen eval/inference mode, no augmentation or normalization refit. All 1440 clean recordings exported: 960 train, 240 val, 240 test. Sequence [T,128] lengths 119–432; pooled [128]. Per-sample files retain ID, label, actor, split, length and padding mask. Pooled NPZ/CSV, provenance, checkpoint, NumPy readers and M1/M5 documentation are provided.

All feature files passed integrity checks, including metadata correspondence, finite shapes, padding, pooling identity, per-file hashes and aggregate alignment. Exported validation pooled features reproduce all 240 Stage 3 validation predictions. Eighteen automated tests passed. Test features were exported without computing test accuracy/F1 or using them for selection.

Not implemented: M1 handcrafted feature extraction/concatenation and representation comparison; M5 multimodal fusion; final test evaluation. M2 pooled output is F_learned, not M1 final F_audio. Audio sequence tokens are not asserted to be frame-aligned to video.


Stage 5/6 follow-up: M2's Stage 5 representations are ready; M1's final F_audio/P_audio remain outside M2 ownership. Reusable Stage 6 delivery packaging, SHA-256 manifest, explicit ownership contract and new-WAV frozen inference code have been implemented. Real packaging and new-WAV inference are intentionally left for the user to run manually. Automated tests use synthetic temporary files only.


## Completed tuning suite — 2026-10-17

Five sequential optimization experiments (15 independent model training/fine-tuning runs) are completed in `runs/tuning_20261017/`. Validation-selected best: round_05 Transformer, local fine-tuning epoch 1, macro-F1 0.53240248 and accuracy 0.54583333. The best checkpoint was reloaded and all 240 validation predictions/metrics were reproduced. Twenty-five automated tests passed. Test set remains unevaluated.

See `docs/TUNING_20261017.md` for all results, rejected configurations, parent checkpoint lineage and exact replay/export commands. Configurations are in `configs/tuning_20261017/`, and the suite contains source snapshots, histories, metrics, predictions, checkpoint hashes and decision notes. Original Stage 3/4/5/6 runs and handoff packages are preserved. The new best model has not yet been re-exported as Stage 4/M1/M5 features; use the guide's commands to create a new feature version.
