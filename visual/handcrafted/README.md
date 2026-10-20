# M3 – Visual Handcrafted Features (2D-DWT) + SVM

## How to run (Yujing)

```bash
uv sync
uv run python -m m3.extract_faces --video-dir "<folder with RAVDESS Video_Speech_Actor_XX>" --out-dir data/faces
uv run python -m m3.dwt_features --faces-dir data/faces --out data/dwt_features.npz
uv run python -m m3.train_svm --features data/dwt_features.npz --out-dir outputs
```

Step 1 can be stopped and restarted; finished videos are skipped. It saves `frame_idx`,
`frame_time_sec` and `video_fps` in each face file. Face files made before this was added can be
backfilled without re-running face detection (safe to re-run; correct files are skipped):

```bash
uv run python -m m3.add_frame_timestamps --video-dir "D:\RAVDESS\Video_Speech" --faces-dir data/faces --modality 01
```

Visual quality check (one 4×4 image of the 16 face frames per video, grouped by emotion;
red border = frame without a detected face):

```bash
uv run python -m m3.contact_sheets --faces-dir data/faces --out-dir data/contact_sheets
```

Tests: `uv run pytest` (set `M3_BASELINE_DIR` to a folder with earlier outputs to also check
that features, probabilities and metrics are unchanged).
`outputs/` (from `m3.train_svm`) is the early interim handcrafted output; it is kept locally only
and is **not** the final delivery. The final visual delivery is
`outputs_visual_comparison/visual_representation_outputs.npz` (see Final Results below).

## Early interim output: `m3_visual_handcrafted.npz` (superseded)

*Superseded by `outputs_augmentation/` (AUG-1) and `outputs_visual_comparison/` (Level-2); kept
for reference only.* The visual handcrafted (DWT) outputs from M3, for developing and testing multimodal fusion.
They do not yet include M4's deep-learned visual features; they will later be replaced by, or
compared with, the final visual output (Level-2 selection of A / B / A+B).

| Field | Shape | Meaning | Used for |
|---|---|---|---|
| `sample_id` | [N] | team sample_id (RAVDESS filename without the modality field) | pairing with audio |
| `split` | [N] | train / val / test from `actor_splits.csv` | |
| `label` | [N] | 0..7, see `class_names` | |
| `F_sequence` | [N, 16, 320] | handcrafted DWT features of each of the 16 frames (before pooling) | Cross-Attention interface testing only (see note) |
| `sequence_length` | [N] | always 16 | |
| `padding_mask` | [N, 16] | True = padding (always False: every clip has 16 frames) | |
| `F_visual_handcrafted` | [N, 320] | temporal mean of `F_sequence` | Feature-Level / Gated Fusion |
| `P_visual_handcrafted` | [N, 8] | DWT-SVM class probabilities | Decision-Level Fusion |
| `class_names` | [8] | neutral, calm, happy, sad, angry, fearful, disgust, surprised | |
| `feature_names` | [320] | `{LL,LH,HL,HH}_r{row}c{col}_{mean,var,energy,entropy,edge}` | |
| `frame_idx` | [N, 16] | frame numbers in the original video; traces where each frame was sampled or supports temporal alignment | optional |
| `frame_time_sec` | [N, 16] | time of each visual frame in the original video, in seconds; can be used if explicit audio-visual temporal alignment is needed | optional |
| `video_fps` | [N] | original video frame rate; verifies `frame_time_sec = frame_idx / video_fps` | optional |

- **About `F_sequence`:** this is a *handcrafted DWT sequence* (320 statistics per frame), not a
  learned representation. It is provided so the Cross-Attention fusion interface can be built and
  tested now. For the final Cross-Attention experiments, M4's **learned visual sequence** (from
  the selected encoder B*) should be used in preference, as planned in the proposal.
- Class order = RAVDESS emotion code 01..08 → label 0..7. **Check it matches the audio files.**
- Features are raw (not standardised). Standardise with train-split statistics before fusion.
- N = 1440 (960 / 240 / 240). Video modality used: 01 (audio-video .mp4).
- **Timing fields** (`frame_idx`, `frame_time_sec`, `video_fps`): the current Feature-Level, Gated
  and Decision-Level Fusion do not require timestamps, and standard Cross-Attention can handle
  audio and video sequences of different lengths directly. The fields are provided for
  traceability and for optional explicit audio-visual temporal alignment.
- SVM hyperparameters were chosen on val Macro-F1; scores are in `metrics.json`. All reported
  metrics and confusion matrices use `model.predict()` (same rule as tuning).
  `P_visual_handcrafted` comes from `predict_proba()` (Platt scaling), so its argmax can differ
  from `predict()` on a few samples.
- The fitted StandardScaler + SVM pipeline is saved as `outputs/dwt_svm.joblib`
  (`joblib.load(...).predict_proba(F_visual_handcrafted)`), e.g. for the demo or external samples.

```python
import numpy as np
d = np.load("m3_visual_handcrafted.npz", allow_pickle=False)
print(d["sample_id"][:3], d["F_sequence"].shape, d["P_visual_handcrafted"].shape)
```

## Method

16 uniformly sampled frames → MediaPipe face detection → rotate so the eyes are level → square
crop → 224×224 → grayscale → centre crop 180×180 (drops most background/hair) → one-level
2D-DWT (Haar) → LL/LH/HL/HH (90×90 each) → each subband split into a 4×4 grid → per cell: mean,
variance, energy (mean of squares), entropy (64-bin histogram), edge (mean Sobel magnitude) →
4 × 16 × 5 = 320 numbers per frame → temporal mean pooling → StandardScaler + RBF-SVM.
Frames with no detected face reuse the nearest detected face box (listed in `data/faces/face_log.csv`).

### Versions (handcrafted feature analysis)

The proposal does not specify over which area the DWT statistics are computed.

| Version | Statistics computed over | Dim / frame | Val Macro-F1 | Test Macro-F1 |
|---|---|---|---|---|
| v1 | whole subband, full 224×224 face | 20 | 0.267 | 0.161 |
| v2 (current) | 4×4 grid, inner 180×180 face | 320 | 0.479 | 0.471 |

v2 differs from v1 in **two** respects at once: (1) an inner 180×180 crop before the DWT and
(2) 4×4 local-grid statistics instead of whole-subband statistics. The controlled ablation below
evaluates the two changes separately.

#### Ablation results (full data, `outputs_ablation/`)

Same face files, splits, label order, seed, SVM search space and train-only scaler for all four;
only `(inner_crop, grid)` changes. Selection uses validation Macro-F1; the test set was evaluated
once, for the selected configuration only.

| Config | inner_crop | grid | Dim / frame | Best C / gamma / class_weight | Train Macro-F1 | Val Macro-F1 | Val Acc | Test Macro-F1 |
|---|---|---|---|---|---|---|---|---|
| A (= v1) | 224 | 1 | 20 | 10 / 0.01 / None | 0.529 | 0.267 | 0.279 | – |
| B | 180 | 1 | 20 | 10 / 0.01 / None | 0.557 | 0.284 | 0.308 | – |
| C | 224 | 4 | 320 | 1 / scale / balanced | 0.885 | 0.376 | 0.375 | – |
| **D (= v2, selected)** | 180 | 4 | 320 | 100 / 0.001 / None | 1.000 | **0.479** | 0.479 | **0.471** (acc 0.454) |

Effect of each change on validation Macro-F1:

| Change | Without the other change | With the other change |
|---|---|---|
| 4×4 grid (vs whole subband) | A → C: +0.109 | B → D: +0.194 |
| 180 crop (vs full 224) | A → B: +0.017 | C → D: +0.102 |

Conclusions:

- **The 4×4 local grid is the main source of the improvement.** It helps with or without the crop.
- **The 180×180 crop alone gives almost nothing** (+0.017 with whole-subband statistics). It helps
  clearly only together with the grid (+0.102), i.e. the two changes interact. This interaction
  is consistent with the hypothesis that, once statistics are computed locally, removing some
  background and hair reduces nuisance variation within individual cells. The ablation supports
  the interaction but does not directly establish this mechanism.
- Exact feature equality was spot-checked on three real samples for both A/v1 and D/v2, and the
  aggregate validation and test scores were reproduced. In addition, the pooled features of all
  1,440 samples matched the stored v1/v2 features within an absolute tolerance of 1e-6, in the
  same sample_id order.
- Caveats: With only one actor-independent split and 240 validation samples, the 0.017
  difference between A and B may reflect sampling variability and should not be interpreted as
  strong evidence of an independent crop effect. D fits the training set perfectly (train
  Macro-F1 1.0), so it overfits, although its test score (0.471) stays close to validation
  (0.479). The v2 design was chosen after seeing v1's validation score, so D's validation score
  may be slightly optimistic. The test score provides the held-out estimate under the fixed
  actor-independent split. However, because the v2 test result had already been observed during
  earlier development, it should be interpreted cautiously rather than treated as a completely
  untouched final estimate.

v2 was selected as the final handcrafted representation according to validation Macro-F1
(0.479 vs 0.267); the test set was not used for this choice. v1 outputs are kept in
`outputs_v1_global20/`; reproduce v1 with `--grid 1 --inner-crop 224`.

### Controlled ablation: inner crop × grid

To separate the two changes, `m3.run_handcrafted_ablation` runs four configurations that differ
**only** in `(inner_crop, grid)`:

| Config | inner_crop | grid | Dim / frame | Note |
|---|---|---|---|---|
| A | 224 | 1 | 20 | = v1 |
| B | 180 | 1 | 20 | crop only |
| C | 224 | 4 | 320 | grid only |
| D | 180 | 4 | 320 | = v2 |

What the three sizes mean:

- **224×224 is the shared base frame size.** `data/faces/*.npz` (16 × 224 × 224 × 3) is used by
  both M3 and M4 and is only read here, never resized, regenerated or overwritten.
- **180×180 is a centre crop inside the M3 handcrafted branch only.** It is taken from the 224×224
  grayscale face inside `dwt_features.frame_features`, just before the DWT.
- **4×4 is the local statistics grid on each DWT subband** (90×90 after a 180 crop, 112×112 without
  crop); the 5 statistics are computed per cell. 90 is not divisible by 4, so cells are 22 or 23
  pixels wide; every cell must be at least 3×3 (checked).
- **None of this requires M4 (Bofei) to change anything:** MobileNetV3 / ResNet18 keep their
  224×224 input.

All four configurations use the same face files, sample_id order, `actor_splits.csv`, label order,
temporal mean pooling, train-only `StandardScaler`, SVM search space (`train_svm.GRID`) and seed.
SVM hyperparameters are chosen on validation Macro-F1; the configuration with the highest
validation Macro-F1 is selected (ties: A, B, C, D order); **only that configuration is evaluated on
the test set, once.** The validation results files contain no test scores.

```bash
uv run python -m m3.run_handcrafted_ablation --faces-dir data/faces --out-dir outputs_ablation --workers 4
# smoke test: first 2 samples of every split x emotion (48 videos)
uv run python -m m3.run_handcrafted_ablation --max-per-class 2 --out-dir outputs_ablation_smoke
```

Results go to `outputs_ablation/` (the runner refuses to write into `outputs/`,
`outputs_v1_global20/` or `data/`, and will not replace earlier ablation results without
`--overwrite`):

| File | Content |
|---|---|
| `ablation_validation_results.csv` / `.json` | per config: crop, grid, dims, best C / gamma / class_weight, train + validation Macro-F1 and accuracy, feature / SVM / total seconds |
| `selected_config.json` | chosen configuration and the selection protocol |
| `test_metrics.json`, `confusion_matrix.csv` | test result of the chosen configuration only |
| `selected_svm.joblib` | fitted StandardScaler + SVM of the chosen configuration |
| `features/{A,B,C,D}.npz` | pooled features per configuration (reload with `--reuse-features`) |

`feature_seconds` is the summed per-video extraction time (CPU time across workers, not wall
time). Each face file is read once and all four configurations are computed from it.

### Train-only data augmentation (DWT-SVM, fixed config D)

`m3.run_handcrafted_augmentation` tests whether augmenting the **training** clips makes the
handcrafted DWT-SVM more robust. Configuration D is fixed (inner crop 180, 4×4 grid, 320 features
per frame, temporal mean pooling); crop and grid are not re-selected.

| Config | Training set | Train samples |
|---|---|---|
| AUG-0 | original train only (no augmentation, = D baseline) | 960 |
| AUG-1 | original train + a horizontally flipped copy of every train clip | 960 + 960 = 1920 |
| AUG-2 | original train + one light combined augmented copy of every train clip | 960 + 960 = 1920 |

AUG-2 parameter ranges (one draw per clip):

| Parameter | Range |
|---|---|
| horizontal flip | probability 0.5 |
| rotation | [−5°, +5°] |
| horizontal / vertical translation | [−3%, +3%] of the frame size |
| scale | [0.97, 1.03] |
| brightness factor | [0.90, 1.10] |
| contrast factor | [0.90, 1.10] (around the clip mean) |

Rules:

- **Augmentation is applied to the train split only.** Validation and test always use the
  original frames and the original D features (`outputs_ablation/features/D.npz`).
- **Clip-consistent augmentation:** each augmented clip gets one parameter set, applied
  identically to all 16 frames (geometry, brightness and contrast), so no artificial temporal
  jitter is introduced. Borders use `cv2.BORDER_REFLECT_101` (no black borders). No vertical
  flip, strong crops, large rotations, CutMix / MixUp, occlusion or blur.
- Augmentation happens on the 224×224 RGB frames, followed by the unchanged D pipeline. The shared
  face files are only read.
- Per-clip parameters come from `sha256("<config>|<sample_id>|<seed>")` (global seed 0), so they
  do not depend on `--workers`, file order or re-runs. Augmented samples are named
  `<sample_id>__aug01` internally and never appear in the 1,440-sample export.
- The StandardScaler is fitted on each configuration's own training set (original + augmented
  train). SVM hyperparameters (same `train_svm.GRID`) and the augmentation configuration are chosen
  on original-validation Macro-F1 only; exact ties prefer AUG-0, then AUG-1, then AUG-2. The test
  set is evaluated once, for the selected configuration only.

```powershell
uv run python -m m3.run_handcrafted_augmentation --faces-dir data/faces --baseline-features outputs_ablation/features/D.npz --out-dir outputs_augmentation --workers 4 --seed 0
# smoke test (separate folder; flow check only, not a result)
uv run python -m m3.run_handcrafted_augmentation --faces-dir data/faces --baseline-features outputs_ablation/features/D.npz --out-dir outputs_augmentation_smoke --workers 2 --seed 0 --max-per-class 1
```

The runner refuses to write into `outputs/`, `outputs_v1_global20/`, `outputs_ablation/` or
`data/`, refuses smoke runs into `outputs_augmentation/`, and does not replace existing results
without `--overwrite`. `--reuse-features` reloads `features/AUG-*.npz` after checking config,
seed, sample_id order, labels, split, dimensions and every augmentation parameter.

| File in `outputs_augmentation/` | Content |
|---|---|
| `augmentation_validation_results.csv` / `.json` | per config: sample counts, best C / gamma / class_weight, train + validation Macro-F1 / accuracy, timing (no test scores) |
| `selected_augmentation.json` | selected config, tie-breaking rule, SVM search space and final parameters |
| `augmentation_config.json` | configuration definitions, parameter ranges, seed derivation, limitations |
| `test_metrics.json`, `confusion_matrix.csv` | test result of the selected configuration only |
| `selected_svm.joblib` | StandardScaler + SVM of the selected configuration |
| `features/AUG-1.npz`, `features/AUG-2.npz` | pooled features of the augmented **train** clips + per-clip parameters |
| `m3_visual_handcrafted_augmented.npz` | candidate export, original 1,440 samples only (see below) |

#### Results (full data, `outputs_augmentation/`, seed 0)

| Config | Train samples | Best C / gamma / class_weight | Train Macro-F1 | Val Macro-F1 | Val Acc | Test Macro-F1 | Test Acc |
|---|---|---|---|---|---|---|---|
| AUG-0 (= D baseline) | 960 | 100 / 0.001 / None | 1.000 | 0.4785 | 0.4792 | – | – |
| **AUG-1 (selected)** | 1920 | 10 / 0.001 / balanced | 0.977 | **0.5290** | 0.5208 | **0.4718** | 0.4667 |
| AUG-2 | 1920 | 10 / scale / balanced | 1.000 | 0.4596 | 0.4792 | – | – |

For reference, the D baseline (= AUG-0) test result from the ablation is Macro-F1 0.4705,
accuracy 0.4542.

- AUG-0 reproduces the D baseline exactly (same hyperparameters, validation Macro-F1 and accuracy).
- AUG-1 (horizontal flip) was selected on validation (+0.050 Macro-F1 over AUG-0). AUG-2 (light
  combined) scored below the baseline on validation.
- On the test set, AUG-1 is almost unchanged from the D baseline: Macro-F1 0.4718 vs 0.4705,
  accuracy 0.4667 vs 0.4542. The validation gain therefore did not carry over clearly to the
  test actors. AUG-1 was chosen out of three configurations (each with its own SVM search) on
  the same 240 validation samples, so its validation score is likely optimistic.
- Per class on test (recall), AUG-1 vs D: calm 0.50 vs 0.44, angry 0.28 vs 0.22, disgust 0.84
  vs 0.72 improved; surprised 0.34 vs 0.47 and fearful 0.22 vs 0.25 dropped; neutral, happy and
  sad recall unchanged (neutral F1 fell from 0.56 to 0.43 because more samples were wrongly
  predicted as neutral).
- The test result is a held-out result under the fixed actor-independent split. Test results of
  the D baseline had already been observed during earlier development, so the comparison should
  be interpreted cautiously.
- **These results do not show that augmentation reliably helps.** Horizontal flipping is a
  reasonable, low-risk choice for the SVM, but its benefit is supported by validation only.
  The no-augmentation D outputs in `outputs/` are kept unchanged and remain available.

**Final status of the three configurations:**

- **AUG-0 is the no-augmentation baseline** (= D; outputs in `outputs/` and `outputs_ablation/`).
- **AUG-1 is the selected configuration and is used from here on.** `selected_svm.joblib`,
  `selected_augmentation.json`, `test_metrics.json`, `confusion_matrix.csv` and
  `m3_visual_handcrafted_augmented.npz` correspond to AUG-1 only. Its test Macro-F1 is
  essentially the same as the baseline, so the benefit of augmentation remains uncertain.
- **AUG-2 is not adopted.** Its validation results (in `augmentation_validation_results.csv` /
  `.json`) and its train-feature cache (`features/AUG-2.npz`) are kept only for experimental
  comparison and reproducibility.

#### Candidate export: `outputs_augmentation/m3_visual_handcrafted_augmented.npz`

Same fields, shapes and order as `outputs/m3_visual_handcrafted.npz` (1,440 original samples:
`sample_id`, `split`, `label`, `F_sequence [1440, 16, 320]`, `sequence_length`, `padding_mask`,
`F_visual_handcrafted [1440, 320]`, `P_visual_handcrafted [1440, 8]`, `class_names`,
`feature_names`, `frame_idx`, `frame_time_sec`, `video_fps`) plus `training_augmentation`
(`"horizontal_flip"`), `augmentation_seed` (0) and `augmentation_config` (JSON).

- `F_sequence` and `F_visual_handcrafted` are still the D features of the **original** clips
  (identical to `outputs/`). Augmentation only changes the SVM training data and therefore
  `P_visual_handcrafted` (probabilities of the AUG-1 SVM, columns in `class_names` order).
- **Limitation for Handcrafted + Deep fusion:** the augmented clips have no matching M4 deep
  features, so they cannot be used for per-sample Handcrafted + Deep concatenation. Fusion must
  still align on the original `sample_id`. Using augmented clips in a combined MLP would require
  M4 to export deep features for exactly the same augmented clips (same parameters per
  sample_id and seed).

## Level-2 Visual Representation Comparison (A / B / A+B)

**Status: formal run completed.** The M4 file (MobileNetV3-Small, frozen backbone) was received,
standardised to `external/m4_visual_deep_outputs.npz` and passed `--validate-only`. The formal
Level-2 run and the actor-grouped 4-fold OOF probabilities are in `outputs_visual_comparison/`.

| Configuration | Classifier | Val Macro-F1 | Test Macro-F1 |
|---|---|---|---|
| **A Handcrafted (selected)** | AUG-1 DWT-SVM | **0.5290** | 0.4718 |
| B Deep | MLP | 0.5048 | 0.3661 |
| A+B Combined | MLP | 0.5188 | 0.4024 |

A was selected on validation Macro-F1. The delivery uses A's `F_visual` and `P_visual_fusion`;
the B and A+B features / probabilities and the deep sequence are also kept in the file.

`m3.run_visual_representation_comparison` compares the three complete visual configurations of
the proposal:

| Config | Representation | Classifier | Training rows |
|---|---|---|---|
| A | handcrafted D feature `F_visual_handcrafted` (320) | RBF SVM — the existing AUG-1 DWT-SVM, **not retrained** | 960 original + 960 flipped = 1920 |
| B | M4 deep feature `F_visual_deep` (D) | StandardScaler + `MLPClassifier` | 960 original |
| A+B | `concatenate([F_visual_handcrafted, F_visual_deep], axis=1)` (320 + D) | StandardScaler + `MLPClassifier` | 960 original |

- **This is a comparison of complete configurations, not a controlled classifier experiment.**
  A uses an SVM, B and A+B use an MLP, and A was trained on more rows, so differences can come from
  the classifier and training scheme as well as from the features. There is no Handcrafted + MLP
  configuration in this experiment.
- A's validation result comes from `outputs_augmentation/augmentation_validation_results.csv`
  (AUG-1 row), its test result and confusion matrix from `outputs_augmentation/test_metrics.json` /
  `confusion_matrix.csv`, its probabilities from `P_visual_handcrafted`, and its model is
  referenced as `outputs_augmentation/selected_svm.joblib` (checked against the probabilities).
- AUG-1 augmented samples are never concatenated with M4 deep features and are never used to
  train B or A+B (the augmented clips have no deep features). The Level-2 feature representations
  still contain only the 1,440 original sample_ids. The read-only AUG-1 cache
  (`outputs_augmentation/features/AUG-1.npz`) is used only to train A's fold-specific SVMs when
  generating actor-grouped OOF train probabilities for downstream fusion.

### M4 input (`external/m4_visual_deep_outputs.npz`)

| Field | Shape | Required |
|---|---|---|
| `sample_id` | [N] | yes (team convention, original recordings only) |
| `split` | [N] | yes (`actor_splits.csv`) |
| `label` | [N] | yes (0–7) |
| `class_names` | [8] | yes (same order: neutral, calm, happy, sad, angry, fearful, disgust, surprised) |
| `F_visual_deep` | [N, D] | yes (pooled clip-level feature of the selected encoder B*) |
| `F_visual_sequence` | [N, 16, D_frame] | optional (passed on for Cross-Attention) |
| `P_visual_deep` | [N, 8] | optional (M4's own classifier; stored as `P_visual_deep_source`) |
| `source_model` | scalar string | required provenance for the formal run: in the NPZ, or pass `--deep-source-model`; if both are given they must be identical (surrounding spaces ignored), otherwise the run stops with a conflict error |
| `source_checkpoint` | scalar string | optional |

If M4 uses other field names, map them explicitly (`--sample-id-key`, `--split-key`,
`--label-key`, `--class-names-key`, `--deep-feature-key`, `--deep-sequence-key`,
`--deep-probability-key`, `--source-model-key`). The program never guesses between similar names;
it lists the actual fields and possible alternatives instead.

**Alignment:** M4 rows are matched to the M3 order **by sample_id, never by row number**. Both
files must contain exactly the same 1,440 unique original sample_ids (no `__aug` / `aug01` /
`augmented` ids); split (960 / 240 / 240), label and class_names must agree for every sample.
Errors name the missing fields, actual shapes, missing / extra sample_id examples and the first
sample_id whose label or split differs. A reorder is reported as `deep_features_reordered`.

### Protocol

- B and A+B use the same pipeline (StandardScaler fitted on train rows only → `MLPClassifier`),
  the same 9 configurations, the same seed and the same 960 train rows: `hidden_layer_sizes` ∈
  {(64,), (128,), (128, 64)} × `alpha` ∈ {1e-4, 1e-3, 1e-2}; fixed `relu`, `adam`,
  `learning_rate_init=1e-3`, `batch_size=32`, `max_iter=500`, `early_stopping=True`
  (`validation_fraction=0.15` taken from the train rows only), `n_iter_no_change=20`.
- Each MLP is chosen on validation Macro-F1 (exact ties: higher validation accuracy, fewer
  parameters, larger alpha, declaration order).
- A / B / A+B is chosen on validation Macro-F1 only; exact ties A > B > A+B (no "near tie" margin).
- Only after that selection is locked are the test rows used: B and A+B are evaluated once each;
  A's test result is reused. Test results never change the selection, hyperparameters, features,
  seed or training scheme. Earlier D / AUG-1 test results had already been observed, so test
  results should be interpreted cautiously.

### Commands

```powershell
# check inputs only (no training; prints WAITING_FOR_M4 if the M4 file is not present)
uv run python -m m3.run_visual_representation_comparison --validate-only
uv run python -m m3.run_visual_representation_comparison --validate-only --deep-features external/m4_visual_deep_outputs.npz
# formal run (after the M4 file passes --validate-only)
uv run python -m m3.run_visual_representation_comparison --deep-features external/m4_visual_deep_outputs.npz --handcrafted outputs_augmentation/m3_visual_handcrafted_augmented.npz --augmentation-dir outputs_augmentation --out-dir outputs_visual_comparison --seed 0
```

The formal run writes only to `outputs_visual_comparison/`, refuses `data/`, `outputs/`,
`outputs_v1_global20/`, `outputs_ablation/` and `outputs_augmentation/`, and does not replace
existing results without `--overwrite`.

### Outputs (`outputs_visual_comparison/`)

| File | Content |
|---|---|
| `validation_results.csv` / `.json` | A row + 9 MLP rows each for B and A+B: classifier, train rows (original / augmented / total), feature dim, hidden sizes, alpha, validation Macro-F1 / accuracy, seed, fit time, selection flags (no test scores) |
| `selected_configs.json` | selected representation, best MLP per representation, A model reference, tie-break rules, MLP search space |
| `test_results.json` | test scores of A (reused), B and A+B, with `test_used_for_selection = false` |
| `confusion_matrix_{handcrafted,deep,combined}.csv` / `.png` | test confusion matrices (raw counts; rows = true label, columns = predicted label; PNGs share one colour range) |
| `representation_comparison.csv` | the three configurations: classifier, feature dim, selected MLP settings (blank for A), validation and test Accuracy / Macro-F1 / Weighted-F1 |
| `classification_report_{handcrafted,deep,combined}.csv` | per-class precision / recall / F1 / support on the 240 test clips, plus accuracy, macro_avg and weighted_avg rows |
| `test_predictions_{handcrafted,deep,combined}.csv` | one row per test clip (same order in all three): true / predicted label, correct, confidence, 8 final-model probabilities (never OOF) |
| `best_deep_mlp.joblib`, `best_combined_mlp.joblib` | locked B and A+B MLPs as dictionary artifacts (see below; no handcrafted MLP) |
| `visual_representation_outputs.npz` | 1,440 original samples: `F_visual_handcrafted`, `F_visual_deep`, `F_visual_combined`, `P_visual_handcrafted` (AUG-1 SVM), `P_visual_deep_mlp`, `P_visual_combined`, `F_visual` / `P_visual` (selected representation), selection metadata, `deep_source_model`; `F_visual_deep_sequence` only if M4 provides it, the handcrafted sequence as `F_visual_handcrafted_sequence` (never presented as a deep sequence) |
| `probability_protocol.json` | OOF folds (fit / holdout actors, sample counts), locked configs, probability sources per split |
| `README.md` | short summary of the formal run |

Probabilities for train-split samples are in-sample predictions and must not be described as
unbiased generalisation results.

The formal run will generate these evaluation files automatically. Model selection still uses
validation Macro-F1 only; Weighted-F1 and all test metrics are reported but never used to select.
`test_results.json` also gives train / validation / test `n_samples`, Accuracy, Macro-F1 and
Weighted-F1 of the locked models (train = ordinary in-sample predictions, not OOF). B and A+B
predict the probability argmax; A keeps the SVM `predict()` rule of its archived AUG-1 results
(Platt probabilities can disagree on a few clips; `argmax_label` / `argmax_matches_prediction`
show where). Outputs are written to a staging folder and published only when complete.

The two MLP joblib files are self-contained **dictionary artifacts**: `artifact["model"]` is the
sklearn Pipeline (StandardScaler -> MLPClassifier); the other keys record the representation
(`B` / `A+B`), input fields and feature dimensions, class order, random seed, the M4
`source_model` / `source_checkpoint`, the selected hidden layers and alpha, and for Combined the
`concatenation_order` (`F_visual_handcrafted`, then `F_visual_deep`) on `concatenation_axis` 1.
No feature arrays are stored. The A model (`outputs_augmentation/selected_svm.joblib`) is still
the existing plain SVM pipeline file, not a dictionary.

```python
import joblib
import numpy as np

deep_artifact = joblib.load("outputs_visual_comparison/best_deep_mlp.joblib")
deep_model = deep_artifact["model"]
deep_probabilities = deep_model.predict_proba(F_visual_deep)

combined_artifact = joblib.load("outputs_visual_comparison/best_combined_mlp.joblib")
combined_model = combined_artifact["model"]
# order from combined_artifact["concatenation_order"]: F_visual_handcrafted, then F_visual_deep
F_visual_combined = np.concatenate(
    [F_visual_handcrafted, F_visual_deep],
    axis=combined_artifact["concatenation_axis"],
)
combined_probabilities = combined_model.predict_proba(F_visual_combined)
```

These evaluation files were generated by the formal run in `outputs_visual_comparison/`.

### Probability protocol for downstream fusion

Final-model probabilities are in-sample on the train split (a model predicting rows it was
trained on), so they must not be used to train a downstream decision-level or gated fusion model.
The run therefore stores two kinds of probabilities:

| Kind | Fields in `visual_representation_outputs.npz` | train rows | val / test rows |
|---|---|---|---|
| Final-model (kept unchanged for compatibility, unimodal analysis, inference on new samples) | `P_visual_handcrafted`, `P_visual_deep_mlp`, `P_visual_combined`, `P_visual` | final model, **in-sample** | final model trained on the full train split |
| **Fusion-safe** (use these for downstream fusion) | `P_visual_handcrafted_fusion`, `P_visual_deep_mlp_fusion`, `P_visual_combined_fusion`, `P_visual_fusion` | actor-grouped 4-fold **OOF** | the same final locked model |

- `P_visual` and `P_visual_fusion` both belong to `selected_representation`. **Downstream
  multimodal training should use `P_visual_fusion`** (or the per-representation `_fusion`
  fields), listed in `probability_fields_for_downstream_fusion`.
- **Folds:** `GroupKFold(n_splits=4)` over the 16 train actors (actor = last field of
  `sample_id`, checked to be 1–24; train / val / test actor sets must be disjoint, 16 / 4 / 4).
  A read-only check also confirms the overall gender balance of the split, counted over unique
  actors (odd ID = male, even = female): train 8 M / 8 F, val 2 M / 2 F, test 2 M / 2 F.
  This applies to the split as a whole; individual GroupKFold folds are not gender-balanced (in
  the real data fold 0 holds out actors 6, 12, 18, 24 and fold 3 holds out 1, 7, 13, 19).
  Each fold holds out 4 actors (240 original train clips); every train clip gets exactly one OOF
  probability. `actor_id` and `oof_fold` (0–3 for train, −1 for val / test) are saved;
  `probability_protocol.json` records each fold's fit / holdout actors and sample counts.
- **A:** each fold fits a clone of the locked AUG-1 SVM (same parameters, no new search) on the
  fit actors' original features **plus only the fit actors' AUG-1 (flipped) features**, matched by
  `source_sample_id` from `outputs_augmentation/features/AUG-1.npz` (read only). The cache must
  contain `config_name == "AUG-1"`, `inner_crop == 180`, `grid == 4` and per-row parameters that
  form a pure horizontal flip (`param_flip` all True; `param_rotation_deg`, `param_translate_x`,
  `param_translate_y` all 0; `param_scale`, `param_brightness`, `param_contrast` all 1; finite,
  one value per row); 960 unique `source_sample_id` equal to the train ids; matching labels,
  train-only split, finite `F_pooled [960, 320]`. `params_json`, `augmentation_ranges` and
  `feature_seconds` are optional. Holdout actors' augmented features never enter that fold. The final A model and
  its `P_visual_handcrafted` are not changed.
- **B / A+B:** each fold builds a fresh StandardScaler + MLP with the hyperparameters already
  locked on validation, fitted on the fit actors' original train rows only (no augmented rows; the
  MLP's early-stopping split also comes from those rows). B and A+B use the same folds.
- Feature-level concatenation still uses the original `F_*` features; augmentation only trains
  A's (fold-specific) SVMs.
- OOF is computed **after** the validation selection is locked. It is not used to select
  anything, it does not replace the validation results, and it is not a Level-2 score.
- Status: the OOF probabilities were generated in the formal run
  (`outputs_visual_comparison/visual_representation_outputs.npz`, protocol in
  `probability_protocol.json`).

## Handcrafted Feature Diagnostics

`m3.diagnose_handcrafted_features` is a **read-only, train-only** health and redundancy check of
the 320 handcrafted DWT features in `outputs_augmentation/m3_visual_handcrafted_augmented.npz`.
All numeric diagnostics use only the 960 original train clips (16 actors); validation / test
rows are read only to check shapes, ids and split counts, and AUG-1 / M4 features are not read.
It does not remove features, train a classifier, select a model or produce a classification
score. All thresholds are fixed in the code and reported in the output.

```powershell
uv run python -m m3.diagnose_handcrafted_features --input outputs_augmentation/m3_visual_handcrafted_augmented.npz --out-dir outputs_handcrafted_diagnostics
```

Outputs in `outputs_handcrafted_diagnostics/` (refuses protected folders; an existing folder is
replaced only with `--overwrite` and only if this tool created it): `README.md`,
`diagnostic_summary.json`, `feature_statistics.csv` (per-feature train statistics and
constant / near-zero flags), `high_correlation_pairs.csv` (all pairs with |r| >= 0.95, upper
triangle), `variance_energy_analysis.csv` (64 subband × cell groups),
`statistics_type_summary.csv`, `subband_summary.csv`.

**Status: the real diagnostic has been run** (input SHA-256
`ce102216922cb6b9790ee2e02d31c2913e2e8f857c1d73041ba84e85b7b54a1b`). Main findings (train only;
diagnostic results, not a classification experiment):

- No NaN / Inf, no constant and no near-zero-variance features.
- Pearson pairs with |r| >= 0.95 / 0.99 / 0.999 / 0.9999: 88 / 55 / 39 / 20. Of these, 48 / 48 /
  39 / 20 are same-cell var–energy pairs; all 20 pairs with |r| >= 0.9999 are var–energy pairs
  (16 HH, 3 LH, 1 HL), involving 40 features. The other high pairs are mostly same-cell LL
  mean–energy and detail-band var/energy–edge pairs. Highest |r| = 0.99999998
  (`HH_r3c0_var`, `HH_r3c0_energy`).
- Variance vs energy: per frame, energy = var + mean² holds in all 64 groups (max |residual|
  1.5e-6). After temporal pooling, pooled_energy = pooled_var + mean_t(frame_mean²) also holds
  in all 64 groups. Because detail-subband cell means are close to 0, pooled var and energy are
  almost perfectly correlated in LH / HL / HH (median |r| ≥ 0.9995) but not in LL (median |r|
  0.35). No pooled var and energy columns are identical.
- By the rule fixed before the run (any constant feature, or ≥ 16 features in pairs with
  |r| >= 0.9999), the result counts as substantial near-duplication. No feature has been removed,
  and test results must not decide any removal. The optional reduced-feature comparison was
  explored locally and was not adopted. The official handcrafted representation remains the
  Full-320 AUG-1 version.

## Final Results / Experiment Summary

1. **Ablation** (`outputs_ablation/`): inner crop × grid. The main gain comes from the 4×4 local
   grid; the 180 crop helps only together with it. Config D (crop 180, grid 4, 320 features per
   frame) was kept.
2. **Augmentation** (`outputs_augmentation/`): AUG-0 / AUG-1 / AUG-2 compared on validation;
   **AUG-1 (horizontal flip) was selected** and is the official Full-320 handcrafted version.
3. **Diagnostics** (`outputs_handcrafted_diagnostics/`, train only): var and energy are almost
   perfectly correlated in the detail subbands (LH / HL / HH); no constant features.
4. **Reduced-272**: explored locally only and **not adopted**; not part of the formal pipeline.
5. **Level-2** (`outputs_visual_comparison/`), selected on validation Macro-F1:

   | Configuration | Classifier | Val Macro-F1 | Test Macro-F1 |
   |---|---|---|---|
   | **A Handcrafted (selected)** | AUG-1 DWT-SVM | **0.5290** | 0.4718 |
   | B Deep | MLP | 0.5048 | 0.3661 |
   | A+B Combined | MLP | 0.5188 | 0.4024 |

6. **Final delivery:** `outputs_visual_comparison/visual_representation_outputs.npz` (A's
   `F_visual` and `P_visual_fusion`; B, A+B and the deep sequence are also included), provided via
   the group's shared drive.

## Repository Contents and Data Availability

- **In Git:** the code (`m3/`, `tests/`), `README.md`, `pyproject.toml`, `uv.lock`,
  `actor_splits.csv`, `.gitignore`, and the lightweight results (CSV, JSON, PNG and README files)
  of four result folders: `outputs_ablation/`, `outputs_augmentation/`,
  `outputs_handcrafted_diagnostics/` and `outputs_visual_comparison/` (including
  `probability_protocol.json`). Poorer ablation and augmentation candidates are kept as evidence
  for the controlled comparisons and model selection.
- **Not in Git (kept locally; large files shared via the group's shared drive):** the face frames
  (`data/faces/`, `data/contact_sheets/`), all feature `.npz` files, all model `.joblib` files, the
  M4 input `external/m4_visual_deep_outputs.npz`, the early interim `outputs/` folder (superseded),
  smoke runs, `outputs_v1_global20/` and the unadopted Reduced-272 files. Paths in this README and
  in the code refer to these files for reproduction; obtain them from the shared storage.
- **Formal visual delivery for fusion:** `outputs_visual_comparison/visual_representation_outputs.npz`
  (shared drive; its `README.md` and `probability_protocol.json` are in Git).
- The official handcrafted representation is **Full-320 AUG-1**.
