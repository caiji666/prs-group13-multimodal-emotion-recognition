#!/usr/bin/env python3
"""Project entry point. Run from any directory with Python 3.11 or 3.12."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT / 'multimodal_fusion/results/m5_final_delivery_20261027'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(verify=False):
    manifest = json.loads((ROOT / 'assets/model_manifest.json').read_text())
    missing, changed = [], []
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file():
            missing.append(item['path'])
        elif path.stat().st_size != item['bytes'] or (verify and sha256(path) != item['sha256']):
            changed.append(item['path'])
    print(f"Model weights: {len(manifest['files']) - len(missing)}/{len(manifest['files'])} present")
    for name, items in [('Missing', missing), ('Checksum mismatch', changed)]:
        if items:
            print(name + ':\n  ' + '\n  '.join(items[:8]))
            if len(items) > 8:
                print(f'  ... and {len(items) - 8} more')
    if missing:
        print('Restore weights with: python scripts/restore_models.py --archive PATH_TO_MODEL_ZIP')
    print('Recorded final fusion runs:', len(list((STUDY / 'runs').glob('*/run.json'))))
    print('Model SHA256 verification:', 'enabled' if verify else 'use --verify for full verification')
    backbone = ROOT / 'demo/assets/wavlm-base-plus/pytorch_model.bin'
    if not backbone.exists():
        print('Prepare the pinned WavLM backbone: python demo/setup_assets.py')
    return 1 if missing or changed or not backbone.exists() else 0


def run(command, cwd=ROOT):
    return subprocess.run([sys.executable, *command], cwd=cwd).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check_parser = sub.add_parser('check', help='Check locally restored model weights')
    check_parser.add_argument('--verify', action='store_true', help='Verify SHA256 of every runtime file')
    demo_parser = sub.add_parser('demo', help='Launch the video prediction interface')
    demo_parser.add_argument('--port', type=int, default=8765)
    extract_parser = sub.add_parser('extract', help='Generate features locally from a video')
    extract_parser.add_argument('--video', type=Path, required=True)
    extract_parser.add_argument('--output', type=Path, required=True)
    extract_parser.add_argument('--audio', choices=['learned', 'handcrafted', 'combined'], default='learned')
    test_parser = sub.add_parser('test', help='Run fusion and demo regression tests')
    test_parser.add_argument('--suite', choices=['core', 'demo', 'all'], default='all')
    sub.add_parser('audit', help='Recompute final metrics and verify all locked checkpoints')
    args = parser.parse_args()
    if args.command == 'check':
        return check(args.verify)
    if args.command == 'demo':
        if check():
            return 1
        print(f'Open http://127.0.0.1:{args.port}', flush=True)
        return run(['-m', 'demo.server', '--port', str(args.port)])
    if args.command == 'extract':
        if check():
            return 1
        from demo.extraction import extract_video
        args.output.mkdir(parents=True, exist_ok=False)
        _, metadata = extract_video(args.video.resolve(), args.output.resolve(), print, audio_representation=args.audio)
        (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        print('Features generated:', args.output / 'extracted_features.npz')
        return 0
    if args.command == 'test':
        if args.suite in ('core', 'all'):
            result = run(['-m', 'unittest', 'discover', '-s', 'tests', '-v'], ROOT / 'multimodal_fusion')
            if result:
                return result
        if args.suite in ('demo', 'all'):
            return run(['-m', 'unittest', 'discover', '-s', 'demo/tests', '-v'])
        return 0
    if args.command == 'audit':
        sys.path.insert(0, str(ROOT / 'multimodal_fusion'))
        from fusion.final_report import validate
        from fusion.paths import study_rows
        validate(STUDY, study_rows(STUDY))
        print('Verified 130 final runs, 260 prediction files, 120 locked checkpoints and 138 gate conditions.')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
