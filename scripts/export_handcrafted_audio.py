"""Generate the original M1 48D fusion-handoff features from speech WAV files."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Allow a partial dataset for extraction checks')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'audio/learned'))
    sys.path.insert(0, str(ROOT / 'demo/vendor/audio/handcrafted'))
    from ravdess_m2.data import manifest
    from src.baseline_v1.feature_pipeline import extract_recording_features
    rows = manifest(args.audio_dir, ROOT / 'multimodal_fusion/configs/actor_splits.csv', full=not args.smoke)
    columns = json.loads((ROOT / 'demo/handcrafted_columns.json').read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'trial_id', 'actor_id', 'emotion_label', 'split', *columns])
        writer.writeheader()
        for row in rows:
            features = extract_recording_features(Path(row['path']))
            if set(features) != set(columns):
                raise ValueError('Extractor output does not match the recorded feature schema')
            writer.writerow(dict(sample_id=Path(row['path']).stem, trial_id=row['sample_id'],
                                 actor_id=row['actor_id'], emotion_label=row['emotion'], split=row['split'], **features))
    print(f'Generated {len(rows)} × 48 handcrafted features: {args.output}')


if __name__ == '__main__':
    main()
