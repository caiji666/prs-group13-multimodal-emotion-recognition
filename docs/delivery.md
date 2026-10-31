# Delivery contents

## Download and run

The [GitHub release](https://github.com/caiji666/prs-group13-multimodal-emotion-recognition/releases/tag/submission-2026-10-31) provides:

| File | Use |
|---|---|
| `Group13_System_Submission_20261031.zip` | Complete system: source, models, final feature data, local WavLM assets, results and available group documents |
| `Group13_Source_20261031.zip` | Tracked repository files, including module models and published result records |
| `Group13_Runtime_Assets_20261031.zip` | Runtime feature data, development cache and WavLM assets for a source checkout |
| `*.zip.sha256` | SHA256 checksums of the corresponding ZIP files |

Extract the complete ZIP and follow `README.md`. A virtual environment is installed on the receiving machine. Every archive contains `PACKAGE_MANIFEST.json` with the source commit and per-file hashes.

## Documents

`submission_documents/` contains the latest locally available group report, project proposal, proposal slides and M5 fusion slides. Documents are copied without content changes. Individual reflections remain separate course submissions.

## Datasets

The runtime bundle supplies all final extracted features used for fusion training/evaluation and validation replay. Original raw RAVDESS videos/WAVs are not bundled. The audio downloader verifies and extracts the official 1,440-recording speech archive:

```bash
python audio/learned/scripts/download_data.py
```

Source and media licensing information is referenced by the module guides. Raw visual extraction requires the paired RAVDESS speech videos and M3's preparation steps in `visual/handcrafted/README.md`. The CREMA-D pilot's fixed download protocol is implemented by `python -m demo.external_eval prepare`; its source revision and per-file checksums are recorded with the published external evaluation.

## Course materials supplied separately

The course handout also calls for raw datasets and a 10–15 minute team presentation recording. Those files are not present in this system package. The supplied presentation decks are the proposal deck and M5 fusion deck. Add the team's presentation recording and any final combined deck to the final Canvas materials, along with the dataset delivery required by the course. See [course requirements](course-requirements.md).

## Build from the repository

```bash
python scripts/package_submission.py --kind source --output dist/Group13_Source_20261031.zip
python scripts/package_submission.py --kind runtime --output dist/Group13_Runtime_Assets_20261031.zip
python scripts/package_submission.py --kind submission --output dist/Group13_System_Submission_20261031.zip
```

The packager uses tracked files plus the explicit runtime manifest. It omits Git metadata, virtual environments, caches, uploaded sessions, private review staging and intermediate document builds.
