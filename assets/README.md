# Model registry

`model_manifest.json` records the paths, sizes and SHA256 checksums of the separately distributed trained model weights. Restore them with:

```bash
python scripts/restore_models.py --archive /path/to/Group13_M1-M5_Model_Weights_20261031.zip
python demo/setup_assets.py
python project.py check --verify
```

The model ZIP is organized into named M1–M5 folders and includes the supplied WavLM backbone and its configuration. The restorer maps `archive_path` names to the original `path` expected by the code; `setup_assets.py` verifies the backbone or downloads it when absent. Feature exports and caches are generated locally using the retained source code and are not distributed in the model ZIP.
