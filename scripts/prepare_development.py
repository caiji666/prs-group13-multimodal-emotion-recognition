"""Build the local train/validation fusion cache from regenerated feature exports."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--visual', type=Path, required=True)
    parser.add_argument('--handcrafted', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'multimodal_fusion/data/final_delivery_20261027')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'multimodal_fusion'))
    import numpy as np
    from fusion.final_inputs import FinalInputs
    reader = FinalInputs(args.audio, args.visual, ROOT / 'multimodal_fusion/configs/actor_splits.csv', args.handcrafted)
    data, audit = reader.load(['train', 'val'])
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'development.npz'
    with target.open('xb') as stream:
        np.savez_compressed(stream, **data)
    (args.output / 'development_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Generated {len(data["labels"])} train/validation samples: {target}')


if __name__ == '__main__':
    main()
