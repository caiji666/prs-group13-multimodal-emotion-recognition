# Delivery verification · 31 October 2026

Validation environment: macOS arm64, Python 3.12. Installed package versions are recorded in the repository-root `requirements-tested.txt`.

| Check | Result | Evidence |
|---|---|---|
| Fusion regression suite | 29 passed | `fusion-tests.txt` |
| Demo/model regression suite | 10 passed | `demo-tests.txt` |
| Local HTTP extraction/inference/download checks | 11 passed | `http-smoke.json` |
| Final fusion metrics and artifacts | 130 records, 260 prediction files, 120 checkpoint hashes and 138 gate conditions verified | Final study `validation_report.json` |
| M2 final tests | 48 records, 11,520 predictions and checkpoint hashes verified | `m2-final-audit.json` |
| Historical M5 records | 30 records, 55 prediction files verified | `m5-history-audit.json` |
| Runtime assets | 1,462 files verified by SHA256 | `assets/runtime_manifest.json` |

Commands are in `docs/operation-guide.md`. The HTTP check uses generated media to exercise upload, FFmpeg decoding, audio/face extraction, inference and downloads. External emotion metrics come from the separate fixed CREMA-D pilot.
