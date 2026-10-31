"""Regression checks against recorded validation predictions; never evaluate test rows."""
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from demo.engine import Engine, CLASSES, RESULTS, ROOT, DEVELOPMENT
from demo.server import Application

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEVELOPMENT.exists():raise unittest.SkipTest('Generate validation features locally to run replay regression checks.')
        cls.engine=Engine();cls.sid=cls.engine.samples()[0]
        cls.data,cls.meta=cls.engine.cached(cls.sid)

    def test_ui_predictions_match_recorded_validation_rows(self):
        for method in ('audio_only','visual_only','concat','gated','cross_attention','cross_bidirectional','audio_residual_aux','lmf_rank4'):
            with self.subTest(method=method):
                output=self.engine.predict(self.data,self.meta,method)
                row=self.engine.row(method)
                with (Path(row['folder'])/'predictions_val.csv').open() as f:
                    saved=next(r for r in csv.DictReader(f) if r['sample_id']==self.sid)
                np.testing.assert_allclose(output['probabilities'],[float(saved['p_'+c]) for c in CLASSES],atol=2e-6,rtol=1e-4)

    def test_decision_terms_exactly_reconstruct_probabilities(self):
        output=self.engine.predict(self.data,self.meta,'decision_tuned')
        exp=output['explanation']
        np.testing.assert_allclose(np.array(exp['audio_term'])+exp['visual_term'],output['probabilities'],atol=1e-7)
        self.assertEqual(exp['alpha_audio'],.6)

    def test_gate_values_and_fixed_target_sensitivity(self):
        r=self.engine.predict(self.data,self.meta,'gated')
        self.assertEqual(len(r['explanation']['values']),64)
        self.assertTrue(all(0<=x<=1 for x in r['explanation']['values']))
        for s in r['sensitivity']:
            self.assertEqual(s['target_class'],r['prediction'])
            self.assertAlmostEqual(s['delta_from_original'],s['target_probability']-r['max_probability'],places=6)

    def test_attention_rows_valid_and_export_does_not_change_prediction(self):
        r=self.engine.predict(self.data,self.meta,'cross_attention')
        att=np.asarray(r['explanation']['heatmap'])
        self.assertEqual(att.shape[1],16)
        np.testing.assert_allclose(att.sum(1),1,atol=2e-6)
        self.assertTrue(r['explanation']['prediction_invariance_checked'])

    def test_unused_visual_intervention_does_not_change_audio_only(self):
        r=self.engine.predict(self.data,self.meta,'audio_only')
        self.assertAlmostEqual(r['sensitivity'][1]['delta_from_original'],0,places=7)

    def test_replay_rejects_test_ids_and_invalid_combinations(self):
        with self.assertRaises(ValueError):self.engine.cached('01-01-01-01-01-02')
        d,m=self.engine.cached(self.sid,'handcrafted')
        with self.assertRaises(ValueError):self.engine.predict(d,m,'concat','handcrafted')

    def test_ui_raw_decision_protocol_never_uses_test(self):
        info=self.engine.raw_visual_alpha()
        self.assertFalse(info['test_used']);self.assertEqual(info['validation_n'],240)
        self.assertIn('visual_only',info['visual_run'])

    def test_raw_hc_control_uses_matching_audio_head_and_independent_visual_head(self):
        d,m=self.engine.cached(self.sid,'handcrafted');m=dict(m,source_kind='uploaded_video')
        result=self.engine.predict(d,m,'audio_only','handcrafted','hc')
        reference=self.engine.head('audio_only',d,'handcrafted','hc')
        np.testing.assert_allclose(result['probabilities'],reference,atol=1e-7)
        self.assertIsNone(result['audio_probabilities'])
        self.assertEqual(len(result['visual_probabilities']),8)

    def test_artifact_hash_mismatch_is_rejected(self):
        engine=Engine();row=engine.row('concat');engine.lock['checkpoint_hashes'][row['run_id']]='bad'
        with self.assertRaisesRegex(ValueError,'SHA256'):engine.load('concat')

    def test_raw_media_does_not_route_by_ravdess_filename(self):
        # The raw path must invoke extraction even when its name mimics a known sample.
        from demo.server import Application
        app=Application();options={'source':'video','audio':'learned','visual':'hc','method':'concat','face_mode':'mediapipe'}
        token,folder=app.create(options)
        try:
            with patch('demo.server.extract_video',side_effect=ValueError('decoder reached')) as extract,patch.object(app.engine,'cached') as cached:
                app.run(token,folder,options,folder/'01-01-01-01-01-09.mp4')
                extract.assert_called_once();cached.assert_not_called()
                self.assertEqual(app.jobs[token]['status'],'failed')
        finally:app.worker.shutdown(wait=True)

if __name__=='__main__':unittest.main()
