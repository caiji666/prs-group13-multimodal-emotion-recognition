"""The upload interface can start without generated dataset feature files."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demo.engine import Engine
from demo.server import Application


class NoPrecomputedFeaturesTests(unittest.TestCase):
    def test_missing_development_cache_exposes_no_replay_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('demo.engine.DEVELOPMENT', Path(directory) / 'not-generated.npz'):
                self.assertEqual(Engine().samples(), [])

    def test_catalog_keeps_video_methods_without_a_replay_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('demo.engine.DEVELOPMENT', Path(directory) / 'not-generated.npz'):
                app = Application()
                try:
                    catalog = app.catalog()
                    self.assertEqual(catalog['samples'], [])
                    self.assertIn('concat', catalog['compatibility']['learned']['hc'])
                    self.assertEqual(catalog['raw_visual'], ['hc'])
                finally:
                    app.worker.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
