"""Restore model weights from the release ZIP, verifying each published SHA256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(archive):
    manifest = json.loads((ROOT / 'assets/model_manifest.json').read_text())
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Archive contains duplicate paths')
        # Validate the complete archive before replacing any project file.
        with tempfile.TemporaryDirectory(prefix='prs-models-') as temporary:
            staged = Path(temporary)
            for item in manifest['files']:
                name = item['path']
                target = (ROOT / name).resolve()
                if not target.is_relative_to(ROOT) or Path(name).is_absolute():
                    raise ValueError(f'Invalid manifest path: {name}')
                path = staged / name
                path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with source.open(item.get('archive_path', name)) as inp, path.open('wb') as out:
                    for block in iter(lambda: inp.read(1024 * 1024), b''):
                        digest.update(block)
                        out.write(block)
                if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
                    raise ValueError(f'Model checksum mismatch: {name}')
            for item in manifest['files']:
                path = ROOT / item['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staged / item['path'], path)
    print(f"Restored {len(manifest['files'])} verified model files to {ROOT}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    restore(parser.parse_args().archive)
