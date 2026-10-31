"""Record provenance and artifact resolution work after ZIP extraction."""
import json
from pathlib import Path
import tempfile
import unittest

from fusion.paths import project_revision, study_rows


class PortableDeliveryTests(unittest.TestCase):
    def test_revision_comes_from_the_extracted_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'PACKAGE_MANIFEST.json').write_text(json.dumps({'git_commit': 'a' * 40}))
            self.assertEqual(project_revision(root), 'a' * 40)

    def test_run_paths_follow_the_moved_study(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            (study / 'runs.json').write_text(json.dumps([
                {'run_id': 'example_run', 'folder': '/original/machine/example_run'}
            ]))
            self.assertEqual(Path(study_rows(study)[0]['folder']), study.resolve() / 'runs/example_run')


if __name__ == '__main__':
    unittest.main()
