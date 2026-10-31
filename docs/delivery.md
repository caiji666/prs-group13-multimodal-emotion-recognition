# Delivery contents

The [lightweight release](https://github.com/caiji666/prs-group13-multimodal-emotion-recognition/releases/tag/submission-2026-10-31-complete-models) provides two downloads:

| File | Contents |
|---|---|
| `Group13_Code_Results_20261031.zip` | Source code, extraction/export scripts, metrics, predictions, figures, guides and group documents |
| `Group13_M1-M5_Model_Weights_20261031.zip` | Trained audio, visual and fusion model weights, restored to the paths in `assets/model_manifest.json` |
| `*.zip.sha256` | ZIP checksums |

Both ZIPs include `PACKAGE_MANIFEST.json` with per-file SHA256 checksums. The model ZIP includes the supplied WavLM backbone and its configuration. Its M1–M5 folders use descriptive filenames, and MODEL_INDEX.json maps them back to the original code paths during restoration. Precomputed dataset features, local development caches, extracted face frames and raw media stay local. Dataset features are produced by the code in [feature-generation.md](feature-generation.md).

## Use the code/results package

Extract it, install dependencies, restore the separate model ZIP, and run `python demo/setup_assets.py`. Follow [the operation guide](operation-guide.md) to launch the interface. A video is processed from its raw input; an existing dataset feature export is not needed for upload prediction.

## Documents

`submission_documents/` contains the locally available group report, project proposal, proposal slides and M5 fusion slides. Individual reflections are submitted separately. Raw datasets, the team presentation recording and the final combined team deck remain separate course materials.

## Build the ZIPs

```bash
python scripts/package_submission.py --kind submission --output dist/Group13_Code_Results_20261031.zip
python scripts/package_submission.py --kind models --output dist/Group13_M1-M5_Model_Weights_20261031.zip
```

Generated artifacts stay ignored in the local workspace. Older feature-inclusive delivery archives are retained only locally and are removed from the public-facing release downloads.
