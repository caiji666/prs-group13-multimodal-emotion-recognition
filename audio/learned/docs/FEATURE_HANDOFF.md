# M2 learned audio features — Stage 4 handoff

This bundle contains the validation-selected **Transformer, epoch 14**, from Stage 3 run `stage3_20261015_194625`. Validation macro-F1: 0.45082615. The checkpoint was selected without test evaluation. Stage 4 uses the frozen checkpoint in evaluation/inference mode, without augmentation, training, normalization refitting or model reselection.

## Contents and dimensions

| Item | Shape / meaning |
|---|---|
| `per_sample/{split}/{sample_id}.npz` | One original speech recording per file |
| `sequence` | float32 `[T, 128]`, contextual encoder output before pooling |
| `pooled` | float32 `[128]`, mean of valid sequence frames, M2 learned vector |
| `length` | int64 scalar, valid encoder frame count |
| `padding_mask` | bool `[T]`, all False because each file is unpadded |
| `sample_id`, `actor_id`, `split`, `label` | Alignment and partition metadata |
| `pooled/{split}.npz` | `pooled [N,128]`, `sample_ids [N]`, `labels [N]`, `actor_ids [N]`, `splits [N]`, `class_names [8]` |
| `pooled/{split}.csv` | Same vectors with metadata and numeric columns `f_000`–`f_127` |
| `manifest.csv` | All sample IDs, labels, splits, lengths, relative feature paths and SHA-256 hashes |
| `metadata.json` | Encoder, checkpoint epoch/hash, split/protocol hashes, dimensions and preprocessing policies |
| `preprocessing.json` | Exact Stage 3 waveform/MFCC configuration |
| `checkpoint/best.pt` | Selected weights, including training-fitted MFCC standardization buffers |
| `provenance/` | Original Stage 3 selection, protocol and completion records |
| `validation_report.json` | Full Stage 4 integrity check; no test accuracy or F1 |
| `handoff_reader.py` | Standalone NumPy reader/alignment/padding helper |

Split sizes: **train 960**, **val 240**, **test 240**. Pooled table shapes are `[960,128]`, `[240,128]`, `[240,128]`. In this completed export, sequence lengths range from **119 to 432 frames** (mean 201.43). All 1,440 original recordings are exported once, including clean, unaugmented training recordings. No augmented duplicate IDs are created. Features are produced in inference mode; dropout is disabled. This is not a new training run.

## Shared IDs, classes and splits

Sample ID preserves the six fields after modality, including leading zeros:
`03-01-06-01-02-01-12.wav` → `01-06-01-02-01-12`.

Class order (official emotion code minus one):
`0 neutral, 1 calm, 2 happy, 3 sad, 4 angry, 5 fearful, 6 disgust, 7 surprise`.

Use the enclosed original `actor_splits.csv`. Match by **sample_id**, then assert matching label and split. Never assume that the row/file order matches another member's output. Downstream model training and scaler fitting must use only train; use val for selection and keep test for final evaluation. Exporting frozen test features does not train on them or measure test performance.

## M1: pooled learned representation

`pooled` is **F_learned**, the output of M2's trained deep encoder. It is not M1's final F_audio. M1 owns handcrafted extraction, F_handcrafted + F_learned concatenation, its classifier and the three representation comparisons. No such fusion is performed in this bundle.

Copy the bundle folder and install NumPy if needed. When your script is beside `handoff_reader.py`:

```python
from pathlib import Path
import numpy as np
from handoff_reader import load_pooled, align_pooled

root = Path('.')  # extracted bundle folder
train = load_pooled(root, 'train')
X_learned = train['pooled']   # [960,128]
y = train['labels']          # [960]
ids = train['sample_ids']

# When a handcrafted table is ready, validate its IDs/labels/splits and
# reorder learned vectors to that table's order. This helper does no fusion:
# aligned = align_pooled(root, 'train', handcrafted_ids,
#                        handcrafted_labels, handcrafted_splits)
```

CSV consumers should read `sample_id` as a string. Do not treat `actor_id` or `label` as feature columns. The numeric learned feature columns are exactly `f_000` through `f_127`.

## M5: sequence representation and padding

```python
from pathlib import Path
from handoff_reader import load_pooled, load_sequence, pad_sequences

root = Path('.')
ids = load_pooled(root, 'train')['sample_ids'][:2]
records = [load_sequence(root, str(sid)) for sid in ids]
batch = pad_sequences(records)
# batch['sequence']: [B,Tmax,128]
# batch['lengths']: [B]
# batch['padding_mask']: [B,Tmax], True = padding/ignored
# For PyTorch: torch.from_numpy(batch['sequence']) and the bool mask.
```

Use audio sequence tokens as Q for the planned cross-attention. The saved length is the post-encoder valid length; this Transformer preserves one token per MFCC frame. MFCC hop is 160 samples at 16 kHz (10 ms). Edge trimming changes the time origin; sequences are not frame-aligned to video. No audio/video frame pairing or timestamp synchronization is asserted here. M5 supplies valid masks for both modalities and must account for any desired temporal alignment separately.

These sequences are encoder representations, not raw MFCCs. They retain the time dimension before global pooling. The pooled vector already averages valid tokens; do not average it again.

## Verify or regenerate

Standalone integrity check (NumPy only):

```python
from handoff_reader import validate_bundle
print(validate_bundle('.'))
```

From the desktop project, reproduce into a **new** directory:

```bash
.venv/bin/python -m ravdess_m2.feature_export export --run runs/stage3_20261015_194625 --output artifacts/stage4_transformer_repeat
```

An existing export directory is rejected. The selected checkpoint's preprocessing and training-fitted normalization are restored automatically. Stage 4 validates shapes, finiteness, pooling identity, checksums, ID uniqueness, split/label correspondence and pooled CSV/NPZ alignment. No task metrics are calculated for test samples. The Stage 3 validation comparison remains the basis for model selection.
