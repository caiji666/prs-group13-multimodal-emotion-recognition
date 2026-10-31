# RAVDESS M2 Audio Deep Learning — Stage 1/2/3/4

Stage 1: official audio-only speech WAV reading; strict supplied actor split; shared sample IDs; mono/resample 16 kHz; edge silence trimming; peak normalization; optional spectral denoising; train-only noise, monotonic time warping and amplitude scaling; variable-length MFCC [T,40].

Stage 2: independent 1D-CNN, BiLSTM, Transformer, each with sequence output, masked global mean pooling, MLP and eight raw logits. Use cross entropy directly on logits; softmax only for probabilities. Stage 3 now implements independent training and validation-based encoder comparison. Formal training has not been launched by the agent; the user starts it manually. Handcrafted feature fusion and multimodal fusion remain outside this project stage.

## macOS setup (Python 3.9–3.12)

```bash
cd ~/Desktop/RAVDESS-M2-Audio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pytest -q
python -m ravdess_m2.dataset_tools validate --data data/raw
python -m ravdess_m2.dataset_tools smoke --data data/raw
python -m ravdess_m2.dataset_tools extract --data data/raw
```

`validate` decodes all WAV files, verifies finite MFCCs and writes manifest/report. `extract` saves clean MFCC sequences for all splits. Augmentation is deliberately online through `SpeechDataset(rows, config, 'train', augment=True)`; validation/test reject augmentation. Use `torch.manual_seed` and a seeded DataLoader generator for reproducibility. Never augment exported M1/M5 features. Configuration is shared by all three models. Stage 1 MFCC files remain unstandardized. Stage 3 fits per-coefficient mean/std using clean training frames only, then stores these as model buffers; the same transform is applied to every split and automatically restored during feature export. No validation/test statistics enter the fit. The supplied config's `augment` value is a suggested training setting; the dataset's explicit `augment` argument controls execution.

## Data and authoritative documents

- `metadata/actor_splits.csv` is copied unchanged from the supplied file. No random split. Train actors: 1,4,5,6,7,10,11,12,13,15,16,18,19,20,23,24. Validation: 9,14,21,22. Test: 2,3,8,17. Expected counts 960/240/240.
- `docs/Project_Proposal_Group13.pdf`, `docs/workdiv.pdf`, `docs/RAVDESS_Sample_ID_Convention.md` are the source references. This project implements Stage 1/2 and the complete Stage 3 training/comparison workflow of M2's broader responsibilities.
- Sample ID removes modality and extension, preserving six two-digit fields, e.g. `03-01-06-01-02-01-12.wav` → `01-06-01-02-01-12`.
- `metadata/label_mapping.json`: official emotion code minus one, ordered neutral/calm/happy/sad/angry/fearful/disgust/surprise. This numeric mapping is an explicit implementation convention; source documents did not supply a team numeric mapping. M1/M5 must confirm/use the same order before fusion.
- Official source: https://zenodo.org/records/1188976 ; audio speech ZIP MD5 `bc696df654c87fed845eb13823edef8a`. Dataset license CC BY-NC-SA 4.0. Only speech, no song. No overwrite of existing user files.

If data are absent, run `python scripts/download_data.py`. Full validation requires 1440 unique recordings (60 per actor). `--allow-partial` is for debugging only. Non-speech audio is excluded; malformed speech names and duplicate IDs fail.

## Encoder / feature API

```python
from ravdess_m2.models import AudioEncoder
model = AudioEncoder('bilstm')  # or cnn / transformer; independent weights
output = model(batch['x'], batch['lengths'])
# x: [B,T,40]; lengths: [B]; collate padding_mask: [B,T]
# output: sequence [B,T_out,128], pooled [B,128], logits [B,8],
#         lengths [B], padding_mask [B,T_out] (True = padding)
```

CNN uses two temporal convolutions + ReLU then max pool, with `T_out=ceil(T/2)`. BiLSTM uses packed valid frames. Transformer projects MFCCs and adds sinusoidal position encodings, then two masked self-attention blocks. All use masked mean global pooling and the same 128→64→8 MLP head. Padding output is zero. MFCC hop is 10 ms; CNN reduces temporal resolution. Trimming and time warping change timing, so sequence indices are not automatically aligned to video frames. M5 cross-attention must consume lengths/masks rather than assume frame-to-frame alignment.

M1 receives `pooled`; M5 can receive `sequence`, `lengths`, `padding_mask`. Join by sample_id and assert matching split and label, never by row order. `pooled` is the M2 learned component only, not M1's final concatenated audio vector.

## Trained feature export

The user has completed Stage 3 run `stage3_20261015_194625`; Transformer epoch 14 was selected by validation macro-F1 (0.45082615). The Stage 4 bundle contains that selected checkpoint. The older generic export command remains available; prefer the Stage 4 handoff workflow below.

```bash
python -m ravdess_m2.dataset_tools export --data data/raw --encoder bilstm --checkpoint runs/my_stage3/bilstm/best.pt --output artifacts/export
```

Each NPZ includes sample_id, actor_id, split, label, sequence, pooled, length, padding_mask, logits, probabilities, class_names. A report records checkpoint/split hashes and preprocessing. Export checks checkpoint preprocessing, split hash and class order against the requested configuration, rejects diagnostic checkpoints and restores trained normalization buffers. Use the original preprocessing configuration. Outputs with the same ID within the specified artifacts directory are replaced when rerunning; use a new `--output` to retain earlier runs.

## Verification

`tests/` covers sample IDs, labels, actor assignments, invalid/duplicate inputs, real audio resampling/MFCC, deterministic val/test, train-only augmentation, batch shapes, padding invariance (including one-frame clips), finite gradients and all three independent models. Synthetic fixtures are test-only and do not replace the supplied split. Full-data validation and model smoke results are recorded separately in `verification/` when run. Backward smoke checks are not model training.

## Stage 3: independent training and validation comparison

Start from the already-installed desktop environment with one command:

```bash
cd "$HOME/Desktop/prs-group13-multimodal-emotion-recognition/audio/learned" && .venv/bin/python -u -m ravdess_m2.train --encoder all --output "runs/stage3_$(date +%Y%m%d_%H%M%S)"
```

This trains CNN, BiLSTM and Transformer **sequentially with independent weights**. Each starts from the same seed and gets the same training sample order and deterministic per-sample/per-epoch augmentation. All candidates use the same actor split, MFCC settings, training-only standardization, MLP head, optimization settings and stopping rule. Shared protocol/source hashes are recorded. Only train/val audio is decoded; test filenames are inspected for dataset integrity, with no test predictions, metrics or model selection.

Default `configs/train.yaml`: CPU, 2 computation threads, batch size 32, up to 30 epochs per model, AdamW (lr 0.001, weight decay 0.0001), cross entropy on logits, gradient clipping at 1.0, ReduceLROnPlateau on validation macro-F1 (factor 0.5, patience 3), early stopping after 7 non-improving epochs. Macro scores always average the same 8 classes; undefined class precision/recall/F1 are zero. The default CPU route is tested. MPS/CUDA flags are available but are not validated on this Mac and may fail deterministic-operator checks. These are reproducible baseline hyperparameters, not tuned optimal settings.

Training regenerates MFCCs from the existing raw WAV files so train-only waveform augmentation is retained. Clean train/val MFCCs are computed once into memory; training uses online augmented MFCCs, validation uses clean cached MFCCs. There is no extra dataset download. The clean MFCC files in `artifacts/mfcc/` are preserved. Early-stopped candidates may complete different epoch counts under the same stopping criterion.

Each output directory must be new; an existing directory is rejected before writing. On each validation improvement (strictly greater macro-F1), `ENCODER/best.pt` is saved; ties retain the earlier epoch. `ENCODER/last.pt` holds the most recent epoch. Checkpoints include model weights and normalization, optimizer/scheduler state, seed/config, RNG state and split/protocol hashes. Automatic resume is not implemented; retain the run and start a separate output directory to restart. An interrupted run leaves saved checkpoints and a failed status; comparison is produced only after all requested models finish.

Output tree:

```text
runs/stage3_TIMESTAMP/
  protocol.json                 # shared settings, source hashes, train/val IDs
  normalization.json            # statistics fitted on clean training frames only
  status.json                   # completion/failure and diagnostic flag
  cnn/                          # also bilstm/ and transformer/
    best.pt
    last.pt
    history.json
    history.csv
    best_val_metrics.json       # accuracy, macro P/R/F1, per-class scores, 8x8 confusion matrix
    best_val_confusion_matrix.csv  # rows=true, columns=predicted; official class order
    best_val_predictions.csv    # sample_id, split, true/predicted labels, 8 probabilities
    summary.json
  comparison.csv
  comparison.json               # selected encoder/checkpoint based on validation macro-F1
```

Exact encoder ties use the fixed order CNN, BiLSTM, Transformer, and all tied encoders are reported. This is a single-seed validation comparison, not evidence of statistical superiority. Test evaluation is deliberately separate and not part of this command. M1/M5 should consume the selected encoder checkpoint after formal training, rather than diagnostic outputs.

To train a single candidate, replace `--encoder all` with `--encoder cnn`, `bilstm` or `transformer`. Only `all` produces the joint comparison. For a manual diagnostic run, append `--smoke` with a new output directory: two epochs on 16 train and 16 val samples, explicitly marked diagnostic, with no best encoder selection and no feature export. Automated tests use tiny synthetic data and do not launch formal RAVDESS training.

## Stage 4: completed learned-feature handoff

`ravdess_m2/feature_export.py` restores the formal Stage 3 validation-selected checkpoint, exports clean train/val/test representations with no augmentation or training, and checks every output. The current selected encoder is Transformer (epoch 14). The bundle is located at `artifacts/stage4_transformer_20261015_194625/` after export. See `docs/FEATURE_HANDOFF.md` for M1/M5 interfaces and standalone NumPy reader examples.

```bash
.venv/bin/python -m ravdess_m2.feature_export validate --output artifacts/stage4_transformer_20261015_194625
```

The bundle provides one NPZ per recording with sequence `[T,128]`, pooled `[128]`, length and mask; split-specific pooled NPZ/CSV; manifest; original actor split; exact preprocessing; selected checkpoint; hashes; verification report; standalone reader. M1 receives the learned vector, not a handcrafted+learned fused vector. M5 receives contextual sequence tokens before pooling. Exporting test features does not evaluate the test set. Stage 4 performs no M1 feature concatenation or M5 multimodal fusion.

## Stage 5/6: M2 handoff contract and reusable delivery/inference code

M2's F_seq and F_learned are ready in Stage 4. Final F_audio and P_audio remain M1 outputs and are explicitly marked pending in the generated contract. `ravdess_m2/delivery.py` validates and packages the selected model, comparison, features, config, standalone inference source and optional ZIP. `ravdess_m2/inference.py` extracts representations from new WAVs and labels its probabilities as M2 learned-only. The user runs delivery/inference manually; no real delivery was generated during implementation.

See `docs/DELIVERY_GUIDE.md` for exact commands, artifact contents and the repeated-experiment workflow.

M5 learned-only testing interface is now generated automatically by the delivery command in `m5_learned_feature_only/`: F_sequence, F_audio_learned_feature_only and P_audio_learned_feature_only, with sample metadata and masks. These use the selected M2 encoder/MLP without handcrafted fusion. M1's existing pooled interface is unchanged. See the Stage 5/6 guide for schemas.


## Completed tuning suite — 2026-10-17

Five sequential optimization experiments (15 independent model training/fine-tuning runs) are completed in `runs/tuning_20261017/`. Validation-selected best: round_05 Transformer, local fine-tuning epoch 1, macro-F1 0.53240248 and accuracy 0.54583333. The best checkpoint was reloaded and all 240 validation predictions/metrics were reproduced. Twenty-five automated tests passed. Test set remains unevaluated.

See `docs/TUNING_20261017.md` for all results, rejected configurations, parent checkpoint lineage and exact replay/export commands. Configurations are in `configs/tuning_20261017/`, and the suite contains source snapshots, histories, metrics, predictions, checkpoint hashes and decision notes. Original Stage 3/4/5/6 runs and handoff packages are preserved. The new best model has not yet been re-exported as Stage 4/M1/M5 features; use the guide's commands to create a new feature version.
