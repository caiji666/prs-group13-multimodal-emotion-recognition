# Model registry

`model_manifest.json` records the paths, sizes and SHA256 checksums of the separately distributed trained model weights. Restore them with:

```bash
python scripts/restore_models.py --archive /path/to/Group13_Model_Weights_20261031.zip
python demo/setup_assets.py
python project.py check --verify
```

The pinned WavLM backbone is downloaded by `setup_assets.py`. Feature exports and caches are generated locally using the retained source code and are not distributed in the model ZIP.
