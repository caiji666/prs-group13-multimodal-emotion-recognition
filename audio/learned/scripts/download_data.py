"""Download official speech WAVs, verify checksum, and extract atomically."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1'
MD5 = 'bc696df654c87fed845eb13823edef8a'


def main():
    target = ROOT / 'data/raw'
    if target.exists():
        raise SystemExit('data/raw already exists; refusing to overwrite. Validate it or move it before retrying.')
    archive = ROOT / 'data/Audio_Speech_Actors_01-24.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix('.zip.partial')
        print('Downloading official RAVDESS speech archive...', flush=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'RAVDESS-M2/1.0'})
        with urllib.request.urlopen(request, timeout=120) as source, partial.open('wb') as output:
            shutil.copyfileobj(source, output)
        partial.rename(archive)
    checksum = hashlib.md5()
    with archive.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != MD5:
        raise SystemExit('Archive checksum mismatch. Remove data/Audio_Speech_Actors_01-24.zip and retry.')
    with tempfile.TemporaryDirectory(prefix='.speech_extract_', dir=archive.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                path = (staging / entry.filename).resolve()
                if not path.is_relative_to(staging.resolve()):
                    raise ValueError('Unsafe archive path')
            source.extractall(staging)
        if len(list(staging.glob('Actor_*/*.wav'))) != 1440:
            raise ValueError('Official speech archive must contain 1440 WAV recordings')
        staging.rename(target)
    print(f'Checksum verified; extracted 1440 WAVs to {target}', flush=True)


if __name__ == '__main__':
    main()
