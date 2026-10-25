# Ten validation-only WavLM continuation experiments

This isolated extension continues the previously selected WavLM Round05 downstream adapter. The pretrained WavLM backbone stays frozen throughout. These experiments fine-tune the adapter, temporal projection, attention/statistics pooling and classifier; they do not update WavLM's 12 Transformer layers.

## Data and selection

Only cache/pooled_index.pt and its referenced training/validation hidden-state files are used: 960 training clips and 240 validation clips, following the shared actor split. All ten experiments use seed42. No test audio, exported test predictions, test metrics or old final-test evaluation files are read by the suite. The old Round05 validation summary supplies the baseline (Accuracy0.8125 / Macro-F1 0.806292282237407). Its validation-only summary predates the separate test evaluation. Previously disclosed test results cannot be retroactively unseen; this new development series relies only on train/val and makes no new independent held-out-test claim.

Before each run, decision_NN.json saves its parent checkpoint hash, exact configuration and changes. The parent is the highest validation Macro-F1 checkpoint among the baseline and completed new experiments, with ties retaining the earlier candidate. Every round trains one distinct configuration. Selection uses Macro-F1, not a post-hoc choice based on accuracy. Maximum65 epochs, early stopping14; the selected epoch can occur early. All new output directories must be unused.

## Methods

1. Freeze projection, attention and statistics pooling; train only MLP. LR0.0003, no mixup/smoothing, dropout0.3.
2. Freeze projection only; LR0.0001, mixup0.1, smoothing0.01, dropout0.3.
3. Train full downstream adapter/head conservatively; LR0.00005, mixup0.1, no smoothing, dropout0.25, weight decay0.01.
4. Add residual frame block LayerNorm→128→64→128, GELU. New final linear initializes to zero, preserving the parent's initial function. LR0.00015, mixup0.1, dropout0.3.
5. Add depthwise temporal Conv1d128→128, kernel3, residual connection, zero initialization. LR0.0001, mixup0.1, dropout0.3. Mask before/after convolution; padded values cannot contaminate valid output.
6. Add residual128→64→128 GELU block before the MLP. LR0.0001, mixup0.2, dropout0.35.
7. Freeze classifier parameters; train temporal/attention/pooling modules. LR0.0001, no mixup, smoothing0.01, dropout0.25. Frozen parameters still use configured train-mode dropout during optimization; evaluation disables it.
8. Add train-only standardized feature Gaussian noise std0.03; mixup0.35, LR0.0001, dropout0.35, weight decay0.03.
9. Focal cross-entropy gamma1; same logits are used for CE and focal probability factor, with the probability factor detached. No feature noise/mixup/smoothing; LR0.00005, dropout0.25.
10. Conservative final continuation; LR0.00002, mixup0.05, dropout0.2, weight decay0.005, no smoothing/no noise, class-balance power0 (uniform weights).

All changes overlay the selected parent's full configuration, so inherited layers remain present. Consult each run's config.json for complete effective settings, including inherited values. AdamW/ReduceLROnPlateau factor0.5 patience4, gradient clip1, batch32. Normalization buffers are inherited from train-only fitted baseline; no refit on validation. Mixup acts on training pooled vectors only.

## Reproduce

From audio/learned, with the original speech-transfer environment, assets and clean cache prepared:

```bash
research/speech_transfer/.venv/bin/python -u -m research.speech_transfer.refinement.suite --output "research/speech_transfer/runs/refinement_$(date +%Y%m%d_%H%M%S)"
```

No virtualenv activation is necessary with the explicit interpreter. The suite retains baseline and previous winners separately; original checkpoints, teammate handoff files and legacy models are never overwritten. The ten continuation records, downstream checkpoints and code are published on main. Raw data, environments, caches and local Deliverables remain excluded.

## Verify and use the winner

```bash
research/speech_transfer/.venv/bin/python -m research.speech_transfer.refinement.verify --output research/speech_transfer/runs/refinement_20261021
```

The independent audit reloads every checkpoint and recomputes all240 validation predictions per run, checks CSV probabilities, metrics, normalization and frozen parameter identity. See suite_results.csv/json, decision_NN.json, per-round checkpoint/config/history/predictions/confusion matrix/provenance, and verification.json under the output folder. Training-time code hashes are in per-round provenance; source snapshots are retained with the completed report.

Use `python -m research.speech_transfer.refinement.inference` with --checkpoint, --audio and --output to load the new architecture and produce NPZ learned features/probabilities. The original inference loader does not know about these extra layers. Output sequence and pooled dimensions remain128; pooled uses attention statistics and learned projection, not simple sequence mean. Old M1/M5 deliveries are not regenerated automatically.

## Shared experiment records

Repository-root `experiment_records/M2_encoder_wavlm_refinement_s42_20261021_01` through `_10` contain schema-v2 records and portable single-run configs. Run `python -m research.speech_transfer.refinement.validate_records` from audio/learned. Per-run checkpoint weights are the exact trained tensors; upstream weights still require the pinned downloader. The training-time git base is 9f14e85, with hashed source snapshots preserving new uncommitted code. Publication includes the earlier independent verification logs; no training was rerun to publish records.
