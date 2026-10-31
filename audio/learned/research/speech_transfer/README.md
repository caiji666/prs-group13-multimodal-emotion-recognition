# Frozen WavLM speech transfer experiments

This isolated research branch adds a frozen pretrained WavLM Base+ backbone and trainable adapters. Existing MFCC CNN, BiLSTM, Transformer, selected model, and teammate delivery formats are unchanged. This extension is published with five shared-format experiment records and downstream weights; upstream assets, cache and environments are downloaded or regenerated locally.

## Recorded results

All five experiments used the shared actor split (960 training, 240 validation), seed 42, and validation Macro-F1 for checkpoint selection. Test audio was not decoded and test metrics remain unmeasured.

| Experiment | Main change | Validation accuracy | Validation Macro-F1 |
|---|---|---:|---:|
| Existing baseline | MFCC Transformer, October 18 Round 04 | 0.545833 | 0.532837 |
| Round 01 | Frozen WavLM layer 12, mean pooling, 128-dimensional adapter | 0.675000 | 0.684081 |
| Round 02 | Replace mean pooling with mean + standard deviation | 0.725000 | 0.727131 |
| Round 03 | Learn mixture weights over layers 6, 9, 12 statistics | 0.787500 | 0.780654 |
| Round 04 | Use strongest layer 6; temporal projection and attentive statistics pooling | 0.808333 | 0.801674 |
| Round 05 | Warm-start Round 04; pooled-feature mixup and regularization refinement | **0.812500** | **0.806292** |

Best local adapter checkpoint: `runs/round_05/best.pt`. It requires the separate frozen upstream weights in `assets/wavlm-base-plus/`. It is not a standalone replacement for the legacy model checkpoint.

See [EXPERIMENTS.md](EXPERIMENTS.md) for changes and limitations; `suite_results.json`, `suite_results.csv`, `best_model.json`, and each run directory contain actual results. Each run saves best weights, history, validation predictions, metrics, confusion matrix, and provenance. No test scores are recorded.

## Setup and reproduce

Run commands from the repository's `audio/learned` directory. Prepare the RAVDESS speech data under `data/raw` using the existing module README first. The shared split file is read from `multimodal_fusion/configs/actor_splits.csv` at the repository root.

```bash
python3 -m venv research/speech_transfer/.venv
research/speech_transfer/.venv/bin/python -m pip install -r research/speech_transfer/requirements.txt
research/speech_transfer/.venv/bin/python -m research.speech_transfer.download
research/speech_transfer/.venv/bin/python -m research.speech_transfer.prepare
research/speech_transfer/.venv/bin/python -u -m research.speech_transfer.suite --output "research/speech_transfer/runs/replay_$(date +%Y%m%d_%H%M%S)"
```

The environment, assets, cache, and recorded run directories already exist on this Mac. Replaying the suite only requires the last command. No environment activation is necessary when using the explicit interpreter. A new output directory is required. The suite resolves warm-start paths to its newly generated checkpoints and chooses the Round 04 layer from Round 03's learned mixture weights. Hardware and library differences can change numerical results; `requirements-research-tested.txt` records this Mac's installed versions.

The pinned official model revision is `4c66d4806a428f2e922ccfa1a962776e232d487b`; the downloader validates its weight SHA-256. The upstream encoder remains frozen throughout; only adapters, pooling, and classifier heads train. The first preparation extracts train/validation features once; subsequent head experiments reuse this cache.

## Single-audio inference

```bash
research/speech_transfer/.venv/bin/python -m research.speech_transfer.inference \
  --checkpoint research/speech_transfer/runs/round_05/best.pt \
  --audio data/raw/Actor_01/03-01-01-01-01-01-01.wav \
  --output research/speech_transfer/runs/example_prediction.npz
```

Use `--external` for non-RAVDESS input. Outputs include class order, logits/probabilities, sample metadata, sequence lengths/mask, `F_sequence`, `F_audio_learned_feature_only`, and `P_audio_learned_feature_only`. Round 04/05 sequence and pooled features have 128 dimensions. Earlier pooled-only adapters retain 768-dimensional frozen sequences and 128-dimensional adapted pooled features. Padding mask uses `True` for padding. Sample IDs retain the existing six-field convention; class 8 is `surprise`.

These learned-only audio outputs are not M1's handcrafted-plus-learned final representation or classifier probabilities. Existing delivery exports have not been replaced. Integration into the legacy export commands would require an explicit loader change; use this branch's inference loader for these checkpoints.

## Validation and source layout

```bash
research/speech_transfer/.venv/bin/python -m pytest research/speech_transfer/tests -q -p no:cacheprovider
```

- `download.py`: pinned upstream asset acquisition and hash validation.
- `prepare.py`: speech decoding, shared split, frozen hidden-state extraction and train/validation cache.
- `models.py`: separate pooled and temporal adapters with MLP heads.
- `run.py`: one training experiment, checkpoint selection and validation outputs.
- `suite.py`: five-experiment replay with dependent warm-start selection.
- `inference.py`: raw WAV inference and learned feature interface.
- `configs/`, `decisions.json`: exact experiment settings and rationale.
- `source_snapshots/`: hashed training-time source snapshots. Later `run.py` additions only check cache provenance before training; the numerical training procedure is unchanged.
- `baseline_reference.json`: hashes of 497 original tracked module/experiment files.
- `verification.json`: independent metrics and raw-audio inference audit. All 240 validation WAVs were decoded again; predictions match cached inference within 6.56e-7 maximum probability difference. All 497 original tracked files remained unchanged.

Model documentation: https://huggingface.co/microsoft/wavlm-base-plus

## Current WavLM M5 handoff and final test evaluation

After model selection was frozen, all 1440 clean speech clips were exported to repository-root `Deliverables/wavlm_round05/m5_learned_feature_only/`, preserving the legacy per_sample/pooled/manifest contract. The adjacent ZIP contains that folder directly at ZIP root. This export reads test audio solely for fixed-model inference. The learned pooled vector is attentive-statistics pooling, not arithmetic sequence mean. Earlier statements above that handoff files were not generated describe the five-experiment completion stage; this new delivery is separate from the old handoff and does not replace the old models.

```bash
research/speech_transfer/.venv/bin/python -u -m research.speech_transfer.export_delivery --output ../../Deliverables/wavlm_round05_NEW
research/speech_transfer/.venv/bin/python -m research.speech_transfer.evaluate_delivery --delivery ../../Deliverables/wavlm_round05_NEW/m5_learned_feature_only
```

Both commands require unused output directories; they do not silently overwrite a completed delivery/evaluation. The exporter does not calculate test scores. The second command was subsequently run with explicit user authorization for final held-out evaluation: test Accuracy 0.8291666666666667 (199/240), Macro-F1 0.8280119634560423, Macro-Precision 0.8628252108779855, Macro-Recall 0.82421875. Metrics, per-class scores, predictions and confusion matrix are saved separately in `Deliverables/wavlm_round05/final_test_evaluation/`. No further training or model selection occurred. Historical validation-only summaries have not been rewritten.

## Published records

Repository-root `experiment_records/M2_encoder_wavlm_s42_20261020_01` through `_05` contain schema-v2 records, downstream weights, portable configs, histories and validation predictions. Round05 additionally contains the locked final test evaluation. Run `python -m research.speech_transfer.validate_records` from audio/learned to verify the new records. The historical MFCC validator continues to validate the original 33 records. Total: 38 M2 records.

## Git publication scope

Only source, configurations, tests, downstream checkpoints and experiment records are versioned. `Deliverables/wavlm_round05/` remains local and is not included in GitHub. References above to that directory describe the local handoff; regenerate it with export_delivery if needed. Final test metrics and prediction CSV are published in `experiment_records/M2_encoder_wavlm_s42_20261020_05/`.

## Ten subsequent continuation experiments

See [refinement/README.md](refinement/README.md) for ten additional experiments based on the original Round05. Current research validation winner is continuation05: Accuracy0.841667 / Macro-F10.831218. All ten test states remain not_run; only the original Round05 has its previously authorized test evaluation. Shared M2 record total is now48, with separate validators for33 MFCC,5 original WavLM and10 continuation records. Existing legacy defaults are unchanged.

## Final test completion —2026-10-26

After development ended, all48 fixed M2 checkpoints received final test records. This supersedes earlier statements of unmeasured test results above; training-time artifacts remain historical. No weights/configurations were updated and no test-based selection was performed. The final selected continuation05 model retains test Accuracy83.3333%, Macro-F10.832127. See repository-root experiment_records/summary.md and shared/M2/final_test_20261026/. Raw datasets and local feature handoffs remain excluded from GitHub.
