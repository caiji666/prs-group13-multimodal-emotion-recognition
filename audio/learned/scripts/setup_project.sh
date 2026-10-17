#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-}" in
  ""|--download-data) ;;
  *) printf 'Usage: bash scripts/setup_project.sh [--download-data]\n' >&2; exit 2 ;;
esac
python3 -c 'import sys; assert (3,9) <= sys.version_info[:2] <= (3,12), "Use Python 3.9-3.12"'
python3 -m venv .venv
if .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3,9))'; then
  .venv/bin/python -m pip install -r requirements-tested.txt
else
  .venv/bin/python -m pip install -r requirements.txt
fi
if [[ "${1:-}" == "--download-data" ]] && [[ ! -d data/raw ]]; then
  .venv/bin/python scripts/download_data.py
fi
if [[ "${1:-}" == "--download-data" ]]; then
  .venv/bin/python -c 'from ravdess_m2.data import manifest; from collections import Counter; rows=manifest("data/raw", "../../multimodal_fusion/configs/actor_splits.csv"); print("Verified speech samples:", len(rows), dict(Counter(r["split"] for r in rows)))'
fi
.venv/bin/python -m pytest -q
printf '\nReady. Use .venv/bin/python from audio/learned; activation is optional.\n'
