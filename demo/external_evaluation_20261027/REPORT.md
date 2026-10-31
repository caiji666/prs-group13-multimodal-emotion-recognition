# External-video generalization pilot — 2026-10-27

## Conclusion / 结论

The raw-video system processes independent CREMA-D recordings successfully, but this pilot does **not** establish reliable external generalization. All **24/24** clips completed extraction and inference; faces were detected in **384/384** sampled frames. The predeclared primary fusion model, concatenation, achieved **9/24 = 37.5% accuracy**, with six-reference-class Macro-F1 **0.3346**. The original M2 audio head achieved **10/24 = 41.7%**, Macro-F1 **0.4076**. Fusion did not consistently outperform this audio baseline.

系统已经能够处理 RAVDESS 之外的真人视频；但“可以输出预测”与“跨数据集识别可靠”不同。这批结果提示当前模型存在明显的迁移局限，不能据此宣称适用于任意真实场景。当前尚无法把领域差异与原始视频预处理复现误差完全区分。

## Frozen protocol

- Source: [official CREMA-D repository](https://github.com/CheyneyComputerScience/CREMA-D), revision `1658cd342dff90010aa843eaeebd53610a08b1dc`. Source README and license are retained alongside this report. The public repository requests that users register usage via its linked form; no personal information was submitted on your behalf.
- Selection was written to [plan.json](plan.json) **before inference**: actors 1001–1004, sentence DFA, intensity XX, all six emotion codes. This is a deterministic convenience sample, not a random representative benchmark. Each emotion has four recordings.
- Reference labels are the official filenames’ **intended/performed emotions**, not crowd-majority ratings or verified internal emotions. Mapping: ANG→angry, DIS→disgust, FEA→fearful, HAP→happy, NEU→neutral, SAD→sad.
- All models, normalizers, checkpoints and seed 42 were fixed. No training, external-label fitting, probability recalibration, threshold search, or RAVDESS test re-evaluation was performed. Model hashes and the validation-only decision-weight grid are in [model_lock.json](model_lock.json).
- The official FLV bytes are checked against the pinned Git LFS SHA256 and decoded directly for scoring. MP4 copies are only for browser upload checks; that separate lossy-transcoded UI example is not mixed into these metrics.
- Audio: original WavLM backbone + M2 adapter, 128D pooled and variable-length 128D sequence. Visual: 16 MediaPipe-aligned faces → Haar DWT 320D per frame and temporal mean. All models use the same extracted inputs.
- **Raw visual probabilities use the M5 visual-only MLP**, because the upstream visual SVM checkpoint has not been delivered. Validation-weighted decision fusion has fixed audio weight **0.70**, learned previously from RAVDESS validation predictions with this MLP. It is a separate UI protocol from the paper’s audio+SVM fusion (0.60).
- Predictions remain **eight-class argmax**. Calm and surprise are not masked and probabilities are not renormalized. They count as errors when predicted for these six-label samples.
- Primary Macro-F1 averages the six reference classes. The secondary eight-class figure assigns zero F1 to the two absent reference classes. These metrics must not be compared directly with the RAVDESS eight-class benchmark as if they were the same evaluation.
- No failures or exclusions. All planned clips contribute to accuracy. Eleven fixed methods yield **264 sample-method predictions**.

## Results

| Frozen pipeline (seed 42) | Correct | Accuracy | Macro-F1, six references | Macro-F1, eight with absent=0 |
|---|---:|---:|---:|---:|
| Original M2 WavLM audio head | 10/24 | 41.67% | 0.4076 | 0.3057 |
| Audio-only MLP | 9/24 | 37.50% | 0.3654 | 0.2740 |
| Visual-only MLP | 6/24 | 25.00% | 0.1828 | 0.1371 |
| Feature concatenation | 9/24 | 37.50% | 0.3346 | 0.2509 |
| Gated fusion | 9/24 | 37.50% | 0.3346 | 0.2509 |
| Cross-attention | 5/24 | 20.83% | 0.1143 | 0.0857 |
| Bidirectional cross-attention | 9/24 | 37.50% | 0.3635 | 0.2726 |
| Audio residual + auxiliary supervision | 10/24 | 41.67% | 0.4055 | 0.3041 |
| Low-rank fusion (LMF) | 11/24 | 45.83% | 0.4175 | 0.3131 |
| Equal-weight decision fusion | 8/24 | 33.33% | 0.3060 | 0.2295 |
| Validation-weighted decision fusion | 9/24 | 37.50% | 0.3251 | 0.2438 |

LMF has the highest descriptive score on this small sample (11 correct); this is **not** a new validation-based selection or evidence of a statistically reliable advantage. One clip changes accuracy by 4.17 percentage points. The 24 clips are clustered within just four actors; no independent-clip confidence interval is claimed.

## What the errors tell us

1. **Detection and transport are working.** Every sample completed real decoding, audio encoding, face detection, handcrafted extraction and inference. This excludes gross pipeline failure for this batch; it does not prove precise parity with teammates’ exported RAVDESS features.
2. **Errors affect both modalities.** Original audio obtains 10 correct and the visual MLP 6. Concatenation and gating both obtain 9, with the same predicted class on every pilot clip. Happy and neutral repeatedly become fearful/sad. Cross-attention predicts sad for 16 clips and calm for five; a normalized attention heatmap cannot establish recognition quality.
3. **High probability can be wrong.** Five incorrect original-audio predictions exceed 0.90 maximum probability; concatenation has two. Probabilities shown in the UI are uncalibrated model outputs, not accuracy estimates.
4. **Domain shift is plausible, not isolated as the sole cause.** Actors, utterances, recording conditions and emotion expression differ from RAVDESS. Label interpretation also differs, and the UI substitutes a visual MLP for the missing original SVM. A raw-RAVDESS parity check is still needed to separate preprocessing mismatch from domain transfer.

## Next experiments, in order

1. Obtain a small set of original RAVDESS videos matching delivered sample IDs. Re-extract with this pipeline; compare pooled/sequence features, original audio probabilities and frozen fusion predictions against the saved exports. Resolve any mismatch before attributing all external errors to generalization.
2. Obtain the final visual SVM artifact and exact preprocessing from M3, then evaluate that **new, explicitly versioned** UI configuration. Keep these MLP-based results unchanged.
3. Freeze a larger cross-dataset protocol covering more actors and all 12 CREMA-D sentences, with documented intended-vs-crowd-label policy and actor-level uncertainty. Do not choose samples based on these predictions. Keep the new evaluation separate from RAVDESS test.
4. If training adaptations, split external actors into adaptation/development/held-out sets first. Compare unchanged checkpoints, calibration, and domain adaptation on the untouched external holdout. Never tune and report on the same 24 clips.
5. Separately assess realistic conditions: spontaneous speech, languages, noise/music, multiple faces and multiple speakers. Add input-quality/unsupported-input handling; validate any rejection rule on development data first. This pilot covers acted English single-speaker clips only.

## Reproduction and artifacts

```bash
demo/.venv/bin/python -m demo.external_eval prepare
demo/.venv/bin/python -m demo.external_eval run
demo/.venv/bin/python -m demo.external_report
```

- [Raw prediction table](predictions.csv): all 264 predictions with reference labels and eight probability columns.
- [Structured summary](summary.json): per-class F1, full 8×8 confusion matrices, failure counts, face coverage and incorrect-prediction probability summaries.
- [Media provenance](media_manifest.json): official URLs, fixed revision, byte counts and SHA256.
- [Team v2 record index](record_index.json): one frozen pipeline × seed per run; `data_kind=external`, separate `metrics_by_split.external` and `predictions_external.csv`. RAVDESS val/test fields remain null because this run evaluates neither.
- [Code snapshot](code_snapshot.zip): the dirty working-tree source used by this pilot, including the frozen fusion implementation; checkpoint hashes are recorded separately. Large weights and raw videos remain local and gitignored.
- Local media: `demo/test-artifacts/crema_d/`; each sample folder contains extracted NPZ, metadata, face crops and per-method explanation JSON. Source videos are not published by this task. CREMA-D ODbL/Database Contents License notices are retained here; follow them if redistributing data or derived databases.

The English UI was verified through a real external MP4 upload, and its gate explanation, modality comparison and uncalibrated-probability labels were inspected. Ten model regression tests and eleven HTTP checks passed after translation.
