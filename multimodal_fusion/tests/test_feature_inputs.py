"""Development-only adapter must work with absent/poisoned test features and probabilities."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fusion import CLASSES
from fusion.data import fit_normalizers
from fusion.feature_inputs import load_development_features, selected_npz_rows


class DevelopmentAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.rng=np.random.default_rng(7)
        actor_map={a:('val' if a in (9,14,21,22) else 'test' if a in (2,3,8,17) else 'train') for a in range(1,25)}
        self.actor=self.root/'actors.csv'
        with self.actor.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['actor_id','split']);w.writerows(actor_map.items())
        self.audio=self.root/'audio.csv';self.seq=self.root/'sequences';self.visual=self.root/'visual.npz'
        ids=[];splits=[];labels=[];visual_sequences=[];audio_rows=[]
        for actor in (1,9,2):
            split=actor_map[actor]
            for label in range(8):
                sid=f'01-{label+1:02d}-01-01-01-{actor:02d}'
                ids.append(sid);splits.append(split);labels.append(label if split!='test' else 999)
                vs=self.rng.normal(size=(16,576)).astype('float32')
                if split=='test':vs[:]=np.nan
                visual_sequences.append(vs)
                row={'sample_id':'03-'+sid,'trial_id':'-'.join(sid.split('-')[1:]),
                     'actor_id':actor,'emotion_label':CLASSES[label],'split':split}
                if split=='test':
                    row.update({f'f{i:03d}':'DO_NOT_PARSE_TEST' for i in range(176)})
                else:
                    sequence=self.rng.normal(size=(3+label,128)).astype('float32')
                    learned=sequence.mean(0)
                    feat=np.concatenate([self.rng.normal(size=48),learned])
                    row.update({f'f{i:03d}':float(v) for i,v in enumerate(feat)})
                    path=self.seq/split/f'{sid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez(path,sample_id=sid,split=split,label=label,actor_id=actor,F_sequence=sequence,
                             length=len(sequence),padding_mask=np.zeros(len(sequence),bool))
                audio_rows.append(row)
        with self.audio.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(audio_rows[0]));w.writeheader();w.writerows(audio_rows)
        seq=np.stack(visual_sequences)
        # Object probabilities must never be decoded (allow_pickle=False).
        np.savez(self.visual,sample_id=np.asarray(ids),split=np.asarray(splits),label=np.asarray(labels),
                 sequence_features=seq,pooled_features=seq.mean(1),probabilities=np.asarray([{}],dtype=object))

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return load_development_features(self.audio,self.seq,self.visual,self.actor,
                                         expected_counts={'train':8,'val':8})

    def test_no_test_files_or_probabilities_loaded(self):
        original=np.load;opened=[]
        def guarded(path,*args,**kwargs):
            self.assertNotIn('test',Path(path).parts)
            opened.append(str(path))
            return original(path,*args,**kwargs)
        with patch('numpy.load',side_effect=guarded):
            d,audit=self.load()
        self.assertEqual(len(d['labels']),16)
        self.assertTrue(np.isfinite(d['visual_sequence']).all())
        self.assertFalse((self.seq/'test').exists())
        self.assertEqual(audit['test_numeric_rows_decoded'],0)
        self.assertFalse(audit['upstream_probabilities_loaded'])
        self.assertEqual(set(d['splits']),{'train','val'})
        self.assertEqual(audit['audio_sequence_files_opened'],16)
        self.assertEqual(len(opened),17)

    def test_pairing_and_train_only_statistics(self):
        d,_=self.load();train=np.flatnonzero(d['splits']=='train')
        stats=fit_normalizers(d,train)
        changed={k:v.copy() for k,v in d.items()}
        val=np.flatnonzero(d['splits']=='val')
        for key in ('audio_pooled','visual_pooled','audio_sequence','visual_sequence'):
            changed[key][val]+=1e5
        self.assertEqual(stats,fit_normalizers(changed,train))
        for sid,l,a in zip(d['sample_ids'],d['labels'],d['actors']):
            self.assertEqual(int(sid.split('-')[1])-1,l)
            self.assertEqual(int(sid[-2:]),a)

    def test_mismatched_actor_split_rejected(self):
        text=self.actor.read_text().replace('9,val','9,train')
        self.actor.write_text(text)
        with self.assertRaisesRegex(ValueError,'split mismatch'):
            self.load()

    def test_selective_row_reader_excludes_poisoned_rows(self):
        values=selected_npz_rows(self.visual,'sequence_features',np.arange(16))
        self.assertTrue(np.isfinite(values).all())
        with self.assertRaises(ValueError):
            selected_npz_rows(self.visual,'probabilities',np.asarray([0]))


if __name__=='__main__':
    unittest.main()
