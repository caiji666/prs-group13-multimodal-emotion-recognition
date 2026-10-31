# Operation guide

Run commands from the repository root. Python 3.11 or 3.12 is supported for the integrated PyTorch system.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/restore_models.py --archive /path/to/Group13_M1-M5_Model_Weights_20261031.zip
python demo/setup_assets.py
python project.py check --verify
```

On Windows PowerShell, use `.venv\Scripts\Activate.ps1`. Download the weights ZIP from the [complete model release](https://github.com/caiji666/prs-group13-multimodal-emotion-recognition/releases/tag/submission-2026-10-31-complete-models). Restoration verifies all model checksums. The model ZIP includes the pinned Microsoft WavLM backbone and configuration for the video interface. `setup_assets.py` verifies the restored files and downloads them only when absent. The backbone remains excluded from Git history.

`requirements-tested.txt` records the integrated macOS / Python 3.12 environment. The original M4 notebooks use a separate TensorFlow environment defined by `visual/visual_deep_learning/requirements.txt`.

## Video prediction

```bash
python project.py demo --port 8765
```

Open **http://127.0.0.1:8765**. Choose a video containing speech, select a compatible method and start analysis. The first prediction loads WavLM; later predictions reuse it. Uploads support clips up to 30 seconds and 100 MB. Use one visible speaker for the clearest face input. Stop with Ctrl+C.

Raw-video prediction generates WavLM audio and MediaPipe-aligned Haar DWT visual features locally. It works without downloaded dataset feature files. Local outputs are written to `demo/sessions/<job-id>/`.

Validation replay becomes available after generating `multimodal_fusion/data/final_delivery_20261027/development.npz` locally. Deep and combined visual representations use that optional cache. The UI disables replay when the cache is absent.

## Feature extraction

```bash
python project.py extract --video /path/to/video.mp4 --output data/my_clip
```

The output contains `extracted_features.npz`, metadata, decoded audio and sampled face images. `--audio handcrafted` and `--audio combined` select the corresponding audio representation. Full dataset export commands are in [feature-generation.md](feature-generation.md).

## Verify results

```bash
python project.py test --suite core
python project.py audit
python audio/learned/scripts/validate_final_tests.py
python scripts/validate_m5_records.py
```

The core suite runs without dataset feature files. The final/M2 audits verify recorded predictions, metrics and downloaded checkpoint hashes. `python project.py test --suite demo` checks saved validation replay when a local development cache exists; otherwise those replay checks are skipped.

The earlier full-system and raw-video verification logs are retained under `docs/verification/`. The lightweight distribution also verifies video inference with no precomputed feature cache.

## Reproduce fusion training

First generate local audio/visual exports and the development cache using the feature generation guide. Then:

```bash
cd multimodal_fusion
python -m fusion.final_study \
  --audio ../features/r05/m5_learned_feature_only \
  --visual ../visual/handcrafted/outputs_visual_comparison/visual_representation_outputs.npz \
  --actors configs/actor_splits.csv \
  --handcrafted ../features/Audio_Handoff/features/handcrafted_features.csv \
  --source-checkpoint ../experiment_records/M2_encoder_wavlm_refinement_s42_20261021_05/best.pt \
  --development data/final_delivery_20261027 \
  --output results/reproduction_final
python -m fusion.final_report --study results/reproduction_final
```

Use a new output folder. The study trains on frozen features, selects with validation Macro-F1, locks model hashes and evaluates the fixed test actors. The exact delivered historical inputs are described by their recorded hashes; newly regenerated exports have their own input audits.

## Build delivery files

```bash
python scripts/package_submission.py --kind submission --output dist/Group13_Code_Results_20261031.zip
python scripts/package_submission.py --kind models --output dist/Group13_M1-M5_Model_Weights_20261031.zip
```

The source/submission packager rejects generated feature payloads and model weights. Each ZIP includes a file manifest and has a SHA256 sidecar.

## Module guides

- [M1 audio features and classifiers](../audio/handcrafted/README.md)
- [M2 audio encoders](../audio/learned/README.md)
- [M3 face/DWT/SVM pipeline](../visual/handcrafted/README.md)
- [M4 deep visual notebooks](../visual/visual_deep_learning/readme.md)
- [M5 fusion study](../multimodal_fusion/README.md)
