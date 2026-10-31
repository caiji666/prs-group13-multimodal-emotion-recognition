"""Resolve experiment artifacts relative to the study that contains them."""
from pathlib import Path


def study_rows(study):
    """Load recorded runs without depending on the original training machine."""
    import json
    study = Path(study).resolve()
    rows = json.loads((study / 'runs.json').read_text())
    for row in rows:
        row['folder'] = str(study / 'runs' / row['run_id'])
    return rows
