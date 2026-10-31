# Delivery verification · 31 October 2026

Validation environment: macOS arm64, Python 3.12. Installed package versions are recorded in the repository-root `requirements-tested.txt`.

| Check | Result | Evidence |
|---|---|---|
| Fusion regression suite | 31 passed | `fusion-tests.txt` |
| Demo/model regression suite | 10 passed | `demo-tests.txt` |
| Local HTTP extraction/inference/download checks | 11 passed | `http-smoke.json` |
| Final fusion metrics and artifacts | 130 records, 260 prediction files, 120 checkpoint hashes and 138 gate conditions verified | Final study `validation_report.json` |
| M2 final tests | 48 records, 11,520 predictions and checkpoint hashes verified | `m2-final-audit.json` |
| Historical M5 records | 30 records, 55 prediction files verified | `m5-history-audit.json` |
| Extracted-package portability | Asset restoration, metric audit and 10 inference checks passed in an independent directory | `portability.json` |
| Earlier full runtime bundle | 1,462 files verified before the distribution was reduced | Historical validation, with feature files now kept local |

Commands are in `docs/operation-guide.md`. The HTTP check uses generated media to exercise upload, FFmpeg decoding, audio/face extraction, inference and downloads. External emotion metrics come from the separate fixed CREMA-D pilot.

The current lightweight package excludes dataset features and weights. Models are restored separately. Raw-video operation is checked without a precomputed development cache; extraction schemas and scripts remain in the repository.

## Lightweight distribution checks

- The source archive contains no generated feature exports or model weights.
- All 200 trained model files restored successfully from the separate model ZIP.
- A clean extracted code package, with no development cache or handcrafted feature-value CSV, passed raw-video HTTP upload, WavLM + DWT extraction, combined-audio inference and file-download checks. Evidence: `lightweight-http.json`.
- Both no-cache interface tests passed; replay regression checks are skipped until the local cache is generated. Evidence: `no-feature-cache-tests.txt`.
- The local regression environment passed all 12 demo tests with its retained cache. Evidence: `lightweight-demo-tests.txt`.
- The handcrafted exporter generated a 48D smoke CSV; the development-cache builder generated 1,200 train/validation rows from local source exports. These outputs remain local.

## Final model supplement

The M1 StandardScaler + LogisticRegression classifier is included in the final model registry (201 files). Its predictions reproduce all 240 delivered validation probability rows, with maximum absolute difference below 4e-16. The measured validation Macro-F1 is 0.388526 and accuracy is 0.4000; no retraining was performed. See `m1-final-model.json`.
