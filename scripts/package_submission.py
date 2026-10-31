"""Build lightweight code/results and separate model ZIPs with checksums."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'Group13_Multimodal_Emotion_Recognition'


def generated_payload(name):
    path = Path(name)
    if path.suffix in ('.pt', '.pth', '.ckpt', '.onnx', '.pkl', '.joblib', '.keras', '.h5', '.hdf5', '.safetensors', '.bin'):
        return True
    if path.suffix in ('.npz', '.npy'):
        return not (path.parent.name == 'gate_diagnostic' and path.name.startswith(('gate_predictions_', 'gate_values_')))
    return path.suffix == '.csv' and 'features' in path.parts



def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def build(kind, output):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    files = set(p for p in tracked if p and (ROOT / p).is_file()) if kind != 'models' else set()
    archive_names = {}
    model_registry = None
    if kind == 'models':
        model_registry = json.loads((ROOT / 'assets/model_manifest.json').read_text())
        weights = model_registry['files'] + model_registry.get('support_files', [])
        for item in weights:
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f"Missing or changed model: {item['path']}")
            files.add(item['path'])
            archive_names[item['path']] = item.get('archive_path', item['path'])
        if len(set(archive_names.values())) != len(archive_names):
            raise ValueError('Model archive names must be unique')
    else:
        forbidden = [name for name in files if generated_payload(name)]
        if forbidden:
            raise ValueError('Generated features or weights cannot be included in the source package: ' + ', '.join(forbidden))
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    prefix = '' if kind == 'models' else PREFIX + '/'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for name in sorted(files):
            path = ROOT / name
            archive_name = archive_names.get(name, name)
            archive.write(path, prefix + archive_name)
            item = {'path': archive_name, 'bytes': path.stat().st_size, 'sha256': digest(path)}
            if kind == 'models':
                item['restore_path'] = name
            entries.append(item)
        if model_registry is not None:
            index = json.dumps(model_registry, indent=2) + '\n'
            archive.writestr('MODEL_INDEX.json', index)
            entries.append({'path': 'MODEL_INDEX.json', 'bytes': len(index.encode()),
                            'sha256': hashlib.sha256(index.encode()).hexdigest()})
            guide = (
                'GROUP 13 MODEL WEIGHTS\n\n'
                'Files are organized and named by M1, M2, M3, M4 and M5.\n'
                'M1: handcrafted audio Logistic Regression and supplied documentation.\n'
                'M2: the supplied CNN, BiLSTM, Transformer, WavLM R05 adapter and complete WavLM backbone.\n'
                'M3: selected 320D handcrafted SVM; 576D deep and 896D combined MLP comparisons.\n'
                'M4: the three GitHub-delivered Keras visual models.\n'
                'M5: local neural, SVM and diagnostic experiment checkpoints, retaining all recorded seeds.\n\n'
                'MODEL_INDEX.json maps readable archive names to the original repository paths.\n'
                'From the code directory run:\n'
                'python scripts/restore_models.py --archive PATH_TO_THIS_ZIP\n\n'
                'The restore command verifies SHA256 and restores the original filenames used by the code.\n'
                'The WavLM backbone and its configuration are included for the video interface.\n'
                'For standalone M2 research scripts, copy the three files in demo/assets/wavlm-base-plus/ '
                'to audio/learned/research/speech_transfer/assets/wavlm-base-plus/.\n'
                'M3 MLP joblibs contain a dictionary; use its model field. The selected SVM is a Pipeline.\n'
                'Precomputed dataset feature files are excluded.\n'
            )
            archive.writestr('README.txt', guide)
            entries.append({'path': 'README.txt', 'bytes': len(guide.encode()),
                            'sha256': hashlib.sha256(guide.encode()).hexdigest()})
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
    parser.add_argument('--kind', choices=['source', 'submission', 'models'], default='submission')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.kind, args.output.resolve())
