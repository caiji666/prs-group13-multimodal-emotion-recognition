"""Readable archive names restore to code paths only after full verification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('restore_models', Path(__file__).resolve().parents[1] / 'restore_models.py')
restore_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_models)


class RestoreModelsTests(unittest.TestCase):
    def make_bundle(self, root, corrupt_second=False):
        payloads = [b'first recorded checkpoint', b'second recorded checkpoint']
        files = [{'path': f'models/{i}/checkpoint.pt', 'archive_path': f'M{i + 1}/Readable_Model.pt',
                  'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                 for i, data in enumerate(payloads)]
        (root / 'assets').mkdir()
        (root / 'assets/model_manifest.json').write_text(json.dumps({'files': files}))
        archive = root / 'weights.zip'
        with zipfile.ZipFile(archive, 'w') as bundle:
            for i, (item, data) in enumerate(zip(files, payloads)):
                bundle.writestr(item['archive_path'], b'changed' if corrupt_second and i == 1 else data)
        return archive, payloads

    def test_readable_names_restore_to_original_code_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive, payloads = self.make_bundle(root)
            with patch.object(restore_models, 'ROOT', root):
                restore_models.restore(archive)
            for i, payload in enumerate(payloads):
                self.assertEqual((root / f'models/{i}/checkpoint.pt').read_bytes(), payload)

    def test_invalid_bundle_does_not_replace_an_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            archive, _ = self.make_bundle(root, corrupt_second=True)
            existing = root / 'models/0/checkpoint.pt'
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b'keep the existing checkpoint')
            with patch.object(restore_models, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    restore_models.restore(archive)
            self.assertEqual(existing.read_bytes(), b'keep the existing checkpoint')


if __name__ == '__main__':
    unittest.main()
