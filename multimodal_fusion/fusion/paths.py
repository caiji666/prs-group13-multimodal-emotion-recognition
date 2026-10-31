"""Resolve experiment artifacts relative to the study that contains them."""
from pathlib import Path


def project_revision(project):
    """Use the checkout revision or the source revision carried in a delivery ZIP."""
    import json
    import subprocess
    project = Path(project)
    if (project / '.git').exists():
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    manifest = project / 'PACKAGE_MANIFEST.json'
    return json.loads(manifest.read_text())['git_commit'] if manifest.exists() else None


def study_rows(study):
    """Load recorded runs without depending on the original training machine."""
    import json
    study = Path(study).resolve()
    rows = json.loads((study / 'runs.json').read_text())
    for row in rows:
        row['folder'] = str(study / 'runs' / row['run_id'])
    return rows
