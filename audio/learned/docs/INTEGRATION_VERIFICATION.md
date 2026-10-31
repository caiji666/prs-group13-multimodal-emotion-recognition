# Integration verification — 2026-10-17

- 27 M2 tests passed, including the shipped checkpoint and shared CSV/JSON agreement.
- Existing fusion padding invariance test passed after changing CLI split defaults.
- Real WAV inference checked for one training actor, one validation actor and one test actor: logits bitwise identical to original round05 Transformer, pooled128, sequence[T,128], eight probabilities summing to one. This is an inference compatibility check, not final test evaluation.
- All 18 migrated historical checkpoints retain bitwise identical model tensors; only class-name metadata changes and an explicit migration record is added.
- NUMBA_CACHE_DIR was placed in a writable temporary directory for sandboxed checks; no dependency installation or original environment modification was needed.
- Full-dataset tuned features have NOT been regenerated. Run scripts/build_handoff.py manually to create new teammate deliveries. Old desktop deliveries remain original-model outputs.
- No retraining, Git commit or push performed during integration.
