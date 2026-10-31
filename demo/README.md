# Video emotion recognition interface

A local browser interface for uploading a short speech video, extracting audio and face features, and comparing trained fusion methods.

```bash
# From the repository root, after environment installation and asset restoration:
python project.py demo --port 8765
```

Open **http://127.0.0.1:8765**. Choose a video up to 30 seconds / 100 MB, or select an existing validation example. Results include the eight-class distribution, audio/visual comparisons and method-specific explanations. JSON and extracted feature NPZ files can be downloaded.

![Interface preview](ui_preview_english.png)

## Available inputs

| Representation | Video upload | Validation replay |
|---|---|---|
| WavLM audio 128D | Yes | Yes |
| Handcrafted audio 48D / combined 176D | Matched audio-only controls | Matched audio-only controls |
| Handcrafted visual 320D | Yes | Yes |
| Deep visual 576D / combined visual 896D | — | Yes |

Compatible methods are listed in the interface. Models use fixed seed42 checkpoints. Raw-video probability fusion uses the original audio head and an M5 visual MLP with its own validation-selected weight (0.70); saved-feature replay uses the delivered original visual probabilities (HC audio weight 0.60).

## Implementation

| Path | Purpose |
|---|---|
| `server.py` | Loopback HTTP service, upload handling, serial inference jobs and result downloads |
| `engine.py` | Matching checkpoint loading, hash verification, replay and explanations |
| `extraction.py` | Audio decoding, pinned WavLM adapter, 16-frame MediaPipe face alignment and Haar DWT |
| `web/` | Browser HTML/CSS/JavaScript, served locally |
| `vendor/manifest.json` | Exact hashes of the original M1/M2/M3 extractor code used here |
| `sessions/` | Uploaded video, decoded audio, sampled faces, extracted features and result JSON |
| `external_evaluation_20261027/` | Fixed 24-video CREMA-D protocol, predictions and metrics |

The extractor matches the pinned upstream feature implementations. WavLM revision and adapter hashes are recorded in the runtime manifest. The visual path uses MediaPipe alignment and Haar DWT; a missing face triggers a reported center crop. See [extraction and explanation definitions](../docs/demo-methods.md).

## Verification

```bash
python project.py test --suite demo
node --check demo/web/app.js
# With the server running:
python -m demo.verify_http
```

The [external-video report](external_evaluation_20261027/REPORT.md) covers 24 official CREMA-D clips, with all 24 processed and faces detected in all 384 sampled frames. The original audio head achieved 10/24 correct; concatenation and gating each achieved 9/24. This small cross-dataset pilot is reported separately from the RAVDESS experiment.

Installation, supported environments and troubleshooting are in the [operation guide](../docs/operation-guide.md).
