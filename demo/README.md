# Video emotion recognition interface

The interface extracts audio and visual features from each uploaded video on the local machine. It uses separately downloaded trained weights and does not require a precomputed dataset feature bundle.

```bash
# After installing the root requirements:
python scripts/restore_models.py --archive /path/to/Group13_Model_Weights_20261031.zip
python demo/setup_assets.py
python project.py demo --port 8765
```

Open **http://127.0.0.1:8765** and select a speech video up to 30 seconds / 100 MB. The interface shows the eight-class distribution, modality comparisons and method-specific explanations. Features and result JSON can be downloaded for the processed clip.

![Interface preview](ui_preview_english.png)

Video extraction supplies WavLM audio (128D) and handcrafted Haar DWT visual features (320D). HC48 and combined176 audio controls are also available. Matched seed42 checkpoints provide concatenation, gate, attention, bidirectional, residual, low-rank and decision methods.

`handcrafted_columns.json` stores only the 48 feature names and order. No feature-value table is needed for handcrafted audio extraction. Validation replay is optional and enabled when a locally generated development cache exists.

| Path | Purpose |
|---|---|
| `server.py` | Local HTTP service, upload handling and inference jobs |
| `engine.py` | Matching checkpoint loading, hash checks and explanations |
| `extraction.py` | Decode raw media and generate features |
| `web/` | Local browser interface |
| `vendor/manifest.json` | Source hashes of the pinned original extractors |
| `sessions/` | Locally generated audio, faces, features and results |
| `external_evaluation_20261027/` | CREMA-D protocol, predictions and metrics |

See [operation instructions](../docs/operation-guide.md), [feature generation](../docs/feature-generation.md) and [explanation definitions](../docs/demo-methods.md). The [external report](external_evaluation_20261027/REPORT.md) retains the fixed 24-video pilot results.
