# Fresh-checkout README verification

Verified on macOS with Python 3.9 on 2026-10-17. Constructed a clean checkout using only `git ls-files --cached --others --exclude-standard`: no original .venv, data, runs, artifacts or deliveries. The local changes are not yet committed/pushed; the eventual push must include all currently Git-visible M2 files plus root .gitignore/.gitattributes and related integration changes.

## Actual checks

1. Ran `bash scripts/setup_project.sh --download-data` in the clean checkout. Installed an independent environment from requirements-tested.txt using pip (cached wheels where available).
2. Downloaded the official speech archive from Zenodo over the network; MD5 matched bc696df654c87fed845eb13823edef8a. Atomic extraction produced 1440 WAVs. Shared split validated as train960/val240/test240.
3. Initialization completed successfully and all 28 tests passed. pip check found no broken dependencies. Shipped selected/baseline/refinement checkpoints loaded without historical runs.
4. Ran latest-refinement configuration for all three encoders with --smoke. Each completed two epochs, diagnostic=true, no selected formal checkpoint and no test metrics. Full formal training/five-round replay was not repeated.
5. Executed README WAV inference example on the downloaded Actor01 neutral recording. Saved sequence[154,128], pooled[128], eight probabilities summing to one; predicted neutral.
6. Executed scripts/build_handoff.py with all downloaded speech recordings. Full 1440-sample export validated; generated complete M1/M5 delivery and ZIP. Checked M1 pooled[960/240/240,128], M5 probability matrices[960/240/240,8], identical IDs, row sums1 and canonical surprise class name.
7. Git inventory includes all required source/config/test/metadata/model files (about16.6MiB, largest file about3.23MiB). Virtual environments, raw data and generated outputs remain ignored.

All generated verification outputs are in an isolated temporary checkout, not the user's formal experiment directories. Existing model results and production handoffs were not overwritten.

## Improvements made

- README now starts with clone, directory selection and initialization, followed by validation, ready-model prediction, fresh/refinement training, tuning and M1/M5 delivery steps.
- Setup checks Python3.9–3.12, validates options, and verifies an existing complete dataset rather than silently trusting data/raw.
- Downloader writes a .partial file, verifies checksum and extracts into staging before exposing data/raw. Incomplete downloads/extraction do not appear as a completed dataset.
- Added .gitattributes to preserve LF for hash-checked sources and split metadata, and mark .pt binaries.

## Limits

Actual verification platform is macOS/Python3.9. README lists supported versions but identical scores across operating systems/dependency versions are not promised. GitHub access, package availability and Zenodo availability are external prerequisites. No commit or push was performed during verification.
