# Operation guide

All commands below run from the repository root unless a `cd` is shown. Paths are relative to the extracted package, so the project can be moved to another directory.

## 1. Environment and assets

Use Python 3.11 or 3.12 with a new virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python project.py check --verify
```

Windows PowerShell uses `.venv\Scripts\Activate.ps1`. `requirements-tested.txt` records the exact Python package versions used for the delivered macOS / Python 3.12 environment; `requirements.txt` supplies the portable dependency ranges.

The integrated environment supports fusion training, audio/video extraction, the web interface and verification. The original M4 notebooks use a separate TensorFlow environment; their dependencies are in `visual/visual_deep_learning/requirements.txt`. M1/M3 also retain their original module environment definitions and detailed experiment commands.

The system ZIP includes all runtime assets. A source checkout requires the runtime ZIP from the GitHub release:

```bash
python scripts/restore_runtime.py --archive /path/to/Group13_Runtime_Assets_20261031.zip
python project.py check --verify
```

The restoration command verifies the full bundle before installing files. The runtime manifest includes the final train/validation cache, 1,440-sample audio handoff, visual feature export, M1 feature-order reference, original audio adapter and pinned WavLM backbone. FFmpeg is supplied by `imageio-ffmpeg`.

## 2. Video prediction interface

```bash
python project.py demo --port 8765
```

Open **http://127.0.0.1:8765**.

1. Choose **Video upload** and a short video containing speech, or choose **Validation replay** and an existing sample.
2. Select audio / visual representations and an available fusion method.
3. Start analysis. The interface decodes the clip, extracts features, loads the matching trained model and displays the eight-class probability distribution.
4. Inspect the method explanation and modality comparisons. Download the result JSON or extracted feature NPZ for the processed clip.

Uploads support clips up to 30 seconds and 100 MB. Use one visible speaker for the clearest face input. The first prediction loads WavLM and takes longer; later predictions reuse the loaded model. Stop the server with Ctrl+C.

Video upload extracts WavLM audio and MediaPipe-aligned Haar DWT visual features. It supports concatenation, gating, attention, bidirectional attention, residual, LMF and decision methods, plus matched audio-only controls. Validation replay additionally supports saved deep and combined visual representations. The original TensorFlow M4 models are delivered in their own module; their preprocessing and weights are not connected to this PyTorch video extraction path.

The interface uses fixed seed42 checkpoints. Raw-video decision fusion combines the original M2 audio head and the M5 visual MLP, with audio weight 0.70 selected on validation predictions. The reported RAVDESS decision result uses delivered visual SVM probabilities and weight 0.60. These configurations have separate records.

Uploads, faces, decoded audio, features and results are written to `demo/sessions/<job-id>/`. Inference stays on the local machine. Restarting the server clears its in-memory job list; saved session files remain on disk.

## 3. Tests and result audit

```bash
python project.py test
python project.py audit
python audio/learned/scripts/validate_final_tests.py
python scripts/validate_m5_records.py
```

`test` runs the fusion and demo regression suites. `audit` recomputes the final 130-run metrics, checks predictions and actor partitions, validates source snapshots and checkpoint hashes, and checks gate diagnostic outputs. It does not train or select a new model.

For the HTTP upload / decoding / prediction checks, start the server and then run in a second terminal:

```bash
python -m demo.verify_http
```

The HTTP check uses a generated two-second video and tone; it verifies the processing chain. The independent CREMA-D results are indexed in [results.md](results.md).

## 4. Reproduce the final fusion study

The runtime bundle includes the inputs needed to rerun the final fusion study. Choose a new output folder to keep the delivered records intact:

```bash
cd multimodal_fusion
python -m fusion.final_study \
  --audio ../features/Audio_dl_features \
  --visual '../features/Video_handcraft/visual_representation_outputs(1).npz' \
  --actors configs/actor_splits.csv \
  --handcrafted ../features/Audio_Handoff/features/handcrafted_features.csv \
  --source-checkpoint data/final_delivery_20261027/upstream_audio_best.pt \
  --development data/final_delivery_20261027 \
  --output results/reproduction_final
python -m fusion.final_report --study results/reproduction_final
```

This trains the declared neural/SVM configurations on frozen features, selects using validation data, locks checkpoints and evaluates the test partition. It reproduces the supplied experiment protocol and uses the same already-observed test actors. Saved checkpoints and predictions support result review without retraining.

## 5. Run individual modules

| Module | Entry point and inputs |
|---|---|
| M1 | [Audio guide](../audio/handcrafted/README.md): feature extraction, preprocessing/classifier comparisons, tuned audio representations and handoff |
| M2 MFCC | [Audio encoder guide](../audio/learned/README.md): setup, dataset download, training and WAV inference |
| M2 WavLM R05 | [Refinement guide](../audio/learned/research/speech_transfer/refinement/README.md): frozen backbone, temporal adapter, inference and export |
| M3 | [Visual guide](../visual/handcrafted/README.md): 16-frame face preparation, DWT-SVM and representation comparisons |
| M4 | [Deep visual guide](../visual/visual_deep_learning/readme.md): three original notebooks, `.keras` models and feature exports |
| M5 | [Fusion guide](../multimodal_fusion/README.md): models, final records, training and input schema |

For example, the selected audio adapter can infer from a WAV file:

```bash
cd audio/learned
python -m research.speech_transfer.refinement.inference \
  --checkpoint ../../experiment_records/M2_encoder_wavlm_refinement_s42_20261021_05/best.pt \
  --audio /path/to/speech.wav --external --output /path/to/prediction.npz
```

This module uses its own WavLM assets directory; follow its downloader guide to prepare the backbone there. The integrated demo's copy is under `demo/assets/wavlm-base-plus/`.

## 6. Build a delivery ZIP

```bash
python scripts/package_submission.py --kind submission --output dist/Group13_System_Submission_20261031.zip
```

`--kind source` builds tracked repository files; `--kind runtime` builds the verified runtime asset bundle. The complete submission package combines both. Each ZIP is checked for archive integrity and has a SHA256 sidecar plus a per-file manifest.

## Troubleshooting

| Symptom | Action |
|---|---|
| Missing runtime assets | Restore the runtime ZIP, then run `python project.py check --verify` |
| Port 8765 is occupied | Use `python project.py demo --port 8766` and open the printed URL |
| No audio track / silent clip | Use a video with an audible speech track |
| No detected face | The interface reports the center-crop fallback; try a front-facing, well-lit single speaker |
| MediaPipe cannot initialize | Run from a normal local terminal with desktop access; initialization errors are surfaced |
| Unsupported representation/method | Select a listed compatible method; checkpoints are specific to their feature dimensions |
| Moving the extracted folder | Run from the new location; artifact paths resolve relative to the project |
