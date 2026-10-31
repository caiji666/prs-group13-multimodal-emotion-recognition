"""Download the exact upstream WavLM weights used by M2; no dataset downloads."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
WEIGHT_SHA = '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'
ASSETS = ROOT / 'assets/wavlm-base-plus'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'preprocessor_config.json', 'pytorch_model.bin'):
        path = ASSETS / name
        if path.exists():
            if name == 'pytorch_model.bin' and sha(path) != WEIGHT_SHA:
                raise ValueError('Existing WavLM checksum does not match the M2 delivery.')
            print('Verified existing', name, flush=True)
            continue
        url = f'https://huggingface.co/microsoft/wavlm-base-plus/resolve/{REVISION}/{name}?download=true&ts={int(time.time())}'
        partial = path.with_suffix(path.suffix + '.partial')
        print('Downloading', name, flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'PRS-M5-LocalDemo/1.0'})
        with urllib.request.urlopen(req, timeout=120) as src, partial.open('wb') as dst:
            count = 0
            while True:
                block = src.read(1024 * 1024)
                if not block: break
                dst.write(block); count += len(block)
                if count % (32 * 1024 * 1024) == 0: print(count // 2**20, 'MiB', flush=True)
        if name == 'pytorch_model.bin' and sha(partial) != WEIGHT_SHA:
            raise ValueError('Downloaded upstream weight checksum mismatch.')
        partial.replace(path)
    (ROOT / 'assets/manifest.json').write_text(json.dumps({
        'model': 'microsoft/wavlm-base-plus', 'revision': REVISION,
        'weights_sha256': sha(ASSETS / 'pytorch_model.bin'),
        'adapter': 'M2_encoder_wavlm_refinement_s42_20261021_05',
        'files': {p.name: sha(p) for p in ASSETS.iterdir() if p.is_file()},
    }, indent=2))
    print('Pinned WavLM assets ready.', flush=True)

if __name__ == '__main__': main()
