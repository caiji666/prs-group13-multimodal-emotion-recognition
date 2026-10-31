# Repository-only workflow

Verified 2026-10-17 in audio/learned's independently installed Python 3.9 virtual environment.

- `.venv/`: independent pip installation from requirements-tested.txt; no old-project sys.path or environment links.
- `data/raw/`: 1440 physical speech WAVs copied once during migration; no symlinks. New clones obtain the official checksum-verified archive through scripts/download_data.py.
- `models/refinement_parents/`: exact CNN round02, BiLSTM round03, Transformer round02 initialization weights, with completed status. best_resolved.yaml points here.
- `models/baseline/`: original baseline checkpoints/status/comparison/metrics, used by the five-experiment replay plan. No ignored historical runs are needed.
- Shared CSV is still the repository's multimodal_fusion/configs/actor_splits.csv.
- 28 tests passed with the new environment. pip check found no broken dependencies.
- Two-epoch diagnostic smoke training completed for all three encoders using only local data/raw, local parent models and local environment. It is explicitly diagnostic and does not replace the selected model or constitute a new formal experiment. Test metrics were not computed.
- Generated feature and delivery schemas remain unchanged. Existing best weights retain their numerical content.

From audio/learned, run:

```bash
# New clone, install dependencies and official speech data
bash scripts/setup_project.sh --download-data
# Latest configuration, three independent warm-start refinements
.venv/bin/python -u -m ravdess_m2.train --encoder all --data data/raw --config configs/tuning_20261017/best_resolved.yaml --output "runs/refinement_$(date +%Y%m%d_%H%M%S)"
# All five original tuning rounds, using the included baseline
.venv/bin/python -u -m ravdess_m2.tuning --plan configs/tuning_20261017/plan.json --output "runs/tuning_$(date +%Y%m%d_%H%M%S)"
# M1/M5 package from selected weights (no training)
.venv/bin/python scripts/build_handoff.py --data data/raw
```

Virtual environments and raw data are ignored by Git and reconstructed on each machine. Baseline/refinement/selected model artifacts are explicitly allowed in Git. No commit/push was performed. Historical metadata may record original experiment locations; those strings are provenance, not runtime dependencies.
