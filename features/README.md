# Final feature handoffs

These runtime files are supplied in `Group13_Runtime_Assets_20261031.zip` and the complete system ZIP. Restore the runtime bundle with `python scripts/restore_runtime.py --archive PATH_TO_ZIP` from the repository root. Expected paths and SHA256 checksums are in `assets/runtime_manifest.json`.

| Path | Contents |
|---|---|
| `Audio_dl_features/` | Selected M2 WavLM R05 export: 1,440 per-sample sequences, pooled features and probabilities; metadata, manifests and checksums |
| `Video_handcraft/visual_representation_outputs(1).npz` | M3's 1,440-sample HC320 / deep576 / combined896 representations and probability exports |
| `Video_handcraft/probability_protocol.json` | Visual probability/OOF protocol |
| `Audio_Handoff/features/handcrafted_features.csv` | Original 48D audio feature values and column order |

The final adapter joins by six-field sample ID and checks labels, actors, splits, dimensions and class order. Earlier local feature exports remain separate from the final runtime manifest.
