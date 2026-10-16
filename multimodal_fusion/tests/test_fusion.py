import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fusion.data import trial_id, load_pair, split_indices, fit_normalizers, normalize
from fusion.demo import generate
from fusion.models import FusionModel
from fusion.experiment import tensor_batch, run, predict, metrics
ROOT = Path(__file__).resolve().parents[1]


class FusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.tmp.name)
        cls.a, cls.v = generate(cls.path/'data')
        cls.d = load_pair(cls.a,cls.v)
        cls.ix,_ = split_indices(cls.d, ROOT/'configs/actors.json')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_trial_pair_and_exclude_song(self):
        self.assertEqual(trial_id('03-01-06-01-02-01-12.wav'),trial_id('02-01-06-01-02-01-12.mp4'))
        self.assertNotEqual(trial_id('03-01-06-01-02-01-12.wav'),trial_id('02-01-06-01-02-02-12.mp4'))
        with self.assertRaises(ValueError):
            trial_id('03-02-06-01-02-01-12.wav')

    def test_shuffled_pair_and_missing_pair(self):
        with np.load(self.v) as z:
            lookup = {s:i for i,s in enumerate(z['sample_ids'])}
            for i in (0,9,100):
                np.testing.assert_array_equal(self.d['visual_pooled'][i],z['pooled'][lookup[self.d['sample_ids'][i]]])
            bad = {k:z[k] if k in ('class_names','data_kind') else z[k][:-1] for k in z.files}
        target = self.path/'missing.npz'
        np.savez(target,**bad)
        with self.assertRaisesRegex(ValueError,'different sample IDs'):
            load_pair(self.a,target)

    def test_actor_leakage(self):
        path = self.path/'bad_split.json'
        path.write_text(json.dumps({'train':[1,2], 'val':[2,3], 'test':[4]}))
        with self.assertRaisesRegex(ValueError,'Actor leakage'):
            split_indices(self.d,path)
        sets = [set(self.d['actors'][ix]) for ix in self.ix.values()]
        self.assertTrue(all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)))

    def test_train_only_statistics(self):
        stats = fit_normalizers(self.d,self.ix['train'])
        changed = {k:x.copy() if isinstance(x,np.ndarray) else x for k,x in self.d.items()}
        changed['audio_pooled'][self.ix['test']] += 1e5
        self.assertEqual(stats,fit_normalizers(changed,self.ix['train']))
        scaled = normalize(self.d,stats)
        np.testing.assert_allclose(scaled['audio_pooled'][self.ix['train']].mean(0),0,atol=1e-5)

    def test_padding_does_not_change_attention(self):
        model = FusionModel('cross_attention',32,40,24,28,hidden=16,dropout=0).eval()
        b = tensor_batch(self.d,np.arange(4),'cpu')
        first = model(b).detach()
        changed = {k:v.clone() for k,v in b.items()}
        for modality in ('audio','visual'):
            x = changed[modality+'_sequence']
            mask = torch.arange(x.shape[1])[None,:] >= changed[modality+'_lengths'][:,None]
            x[mask] = 1e4
        torch.testing.assert_close(first,model(changed))
        extended = {k:v.clone() for k,v in b.items()}
        for modality in ('audio','visual'):
            x = extended[modality+'_sequence']
            extended[modality+'_sequence'] = torch.cat([x,torch.full((4,3,x.shape[2]),1e4)],dim=1)
        torch.testing.assert_close(first,model(extended))

    def test_malformed_probabilities(self):
        with np.load(self.a) as z:
            bad = {k:z[k] for k in z.files}
        bad['probs'][0] = 1.
        path = self.path/'invalid.npz'
        np.savez(path,**bad)
        with self.assertRaisesRegex(ValueError,'probabilities'):
            load_pair(path,self.v)

    def test_checkpoint_roundtrip(self):
        output = self.path/'run'
        report = run(self.a,self.v,ROOT/'configs/actors.json',output,
                     ['decision','concat','gated','cross_attention'],epochs=2,hidden=16,synthetic=True)
        for name in ('concat','gated','cross_attention','decision_tuned'):
            ext = '.json' if name.startswith('decision') else '.pt'
            probs = predict(output/'checkpoints'/f'{name}{ext}',self.a,self.v,output/f'{name}_pred.json')
            np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
            actual = metrics(self.d['labels'][self.ix['test']],probs[self.ix['test']])
            self.assertEqual(actual['confusion_matrix'],report['models'][name]['test']['confusion_matrix'])
        self.assertTrue(report['synthetic'])
        self.assertEqual(report['counts'],{'train':960,'val':240,'test':240})


if __name__ == '__main__':
    unittest.main()
