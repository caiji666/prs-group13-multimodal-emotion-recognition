"""Build source, runtime and complete system ZIPs with per-file checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'Group13_Multimodal_Emotion_Recognition'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(kind, output):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    runtime = json.loads((ROOT / 'assets/runtime_manifest.json').read_text())['files']
    files = set(p for p in tracked if p and (ROOT / p).is_file()) if kind != 'runtime' else set()
    if kind != 'source':
        for item in runtime:
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f"Missing or changed runtime asset: {item['path']}")
            files.add(item['path'])
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    prefix = '' if kind == 'runtime' else PREFIX + '/'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in sorted(files):
            path = ROOT / name
            archive.write(path, prefix + name)
            entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)})
        manifest = {'kind': kind, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    'repository': 'https://github.com/caiji666/prs-group13-multimodal-emotion-recognition', 'files': entries}
        archive.writestr(prefix + 'PACKAGE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP integrity check failed: {bad}')
    sha = digest(output)
    output.with_suffix(output.suffix + '.sha256').write_text(f'{sha}  {output.name}\n')
    print(f'{output}: {len(entries)} files, {output.stat().st_size / 2**20:.1f} MiB, SHA256 {sha}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['source', 'runtime', 'submission'], default='submission')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.kind, args.output.resolve())
