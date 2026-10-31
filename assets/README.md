# Runtime asset registry

`runtime_manifest.json` lists every runtime file, its size and SHA256. The complete system ZIP includes these files. For a source checkout, restore `Group13_Runtime_Assets_20261031.zip` using `python scripts/restore_runtime.py --archive PATH_TO_ZIP`.

The bundle contains final audio/visual feature handoffs, the final train/validation cache, the original M2 adapter and the pinned Microsoft WavLM backbone. All final M5 trainable checkpoints and gate diagnostics are versioned with the study under `multimodal_fusion/results/m5_final_delivery_20261027/`.

Use `python project.py check --verify` to check the complete runtime bundle, and `python project.py audit` to verify the published final experiment results.
