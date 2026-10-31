# Multimodal fusion · M5

Frozen audio and visual representations feed decision-level, concatenation, gated and cross-attention classifiers. Additional comparisons include bidirectional attention, frozen-audio residual correction, auxiliary supervision, low-rank fusion and SVM feature/kernel fusion.

## Final study

The delivered study uses WavLM R05 audio (128D) and handcrafted visual features (320D), with fixed actor partitions and three seeds. It contains 130 records with separately downloadable checkpoints, predictions, histories, source snapshots and evaluation locks.

- [Result overview](../docs/results.md)
- [All 130 configurations](results/m5_final_delivery_20261027/summary.md)
- [Study plan](results/m5_final_delivery_20261027/plan.json)
- [Checkpoint and selection lock](results/m5_final_delivery_20261027/evaluation_lock.json)
- [Operation and reproduction commands](../docs/operation-guide.md)
- [Shared evaluation protocol](../docs/protocol.md)

From the repository root:

```bash
python project.py test --suite core
python project.py audit
python project.py demo
```

## Code guide

| File | Purpose |
|---|---|
| `fusion/data.py` | Feature validation, sample alignment, actor partitions and train-fitted normalization |
| `fusion/models.py` | Concatenation, gate and cross-attention implementations |
| `fusion/final_inputs.py` | Final audio/visual handoff loading and consistency checks |
| `fusion/final_models.py` | Final-study model registry and original frozen audio head |
| `fusion/final_study.py` | Declared training configurations, validation selection, checkpoint lock and test evaluation |
| `fusion/final_report.py` | Metric audit, aggregate tables and figures |
| `fusion/structure_models.py` | Attention and audio-residual comparison architectures |
| `fusion/literature_models.py` | Low-rank fusion and contrastive/auxiliary variants |
| `fusion/paths.py` | Study-relative artifact resolution for moved checkouts |
| `fusion/study*.py`, `structure_study.py`, `literature_study.py` | Earlier feature-version studies and diagnostics |
| `configs/actor_splits.csv` | Shared 16 / 4 / 4 actor partition |
| `tests/` | Alignment, normalizers, padding, checkpoint and model regression checks |

## Generic feature interface

The general-purpose CLI accepts one NPZ per modality with `sample_ids`, `labels`, `actors`, `class_names`, `pooled` and `probs`. Optional `sequence` and `lengths` supply time sequences. IDs are unique six-field speech trial IDs; probabilities use the shared eight-class order. See [`examples/export_features.py`](examples/export_features.py).

```bash
cd multimodal_fusion
python -m fusion validate --audio data/audio.npz --visual data/visual.npz --split configs/actors_m2.json
python -m fusion train --audio data/audio.npz --visual data/visual.npz \
  --split configs/actors_m2.json --output results/new_study --epochs 30
```

Generate the final handoffs locally using [the feature guide](../docs/feature-generation.md). The final handoffs use their richer source schemas and are loaded through `fusion.final_inputs`; use the final-study command in the operation guide for the delivered experiment.

A self-contained generated-data smoke run is also available:

```bash
python -m fusion demo --output results/generated_smoke --epochs 2
```

Generated smoke data checks the implementation. RAVDESS results are indexed separately above. Earlier feature experiments and development notes are preserved in [the archive](../docs/archive/fusion-development-notes.md) and the [shared record index](../experiment_records/summary.md).
