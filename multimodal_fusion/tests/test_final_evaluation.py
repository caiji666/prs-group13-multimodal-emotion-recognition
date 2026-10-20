import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from fusion import CLASSES
from fusion.final_evaluation import load_test_features, select_alpha


class FinalEvaluationTests(unittest.TestCase):
    def test_equal_validation_scores_favor_equal_weight(self):
        y=np.arange(8);p=np.eye(8,dtype=np.float32)
        alpha,grid=select_alpha(y,p,p)
        self.assertEqual(alpha,.5)
        self.assertEqual(len(grid),21)

    def test_weight_search_uses_supplied_validation_predictions(self):
        y=np.arange(8);p=np.eye(8,dtype=np.float32);q=np.roll(p,1,axis=1)
        alpha,_=select_alpha(y,p,q)
        self.assertGreater(alpha,.5)
        # This function takes no test data: its objective is fixed entirely by these labels.
        alpha2,_=select_alpha(y,q,p)
        self.assertLess(alpha2,.5)

    def test_test_loader_ignores_poisoned_development_features_and_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actors=root/'actors.csv';audio=root/'audio.csv';seqdir=root/'sequences'
            with actors.open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['actor_id','split'])
                w.writerows((a,'test' if a==2 else 'val' if a==9 else 'train') for a in range(1,25))
            ids=[];labels=[];visual=[];rows=[]
            for label in range(8):
                sid=f'01-{label+1:02d}-01-01-01-02';ids.append(sid);labels.append(label)
                visual.append(np.full((16,576),label+.1,np.float32))
                aseq=np.full((3,128),label+.25,np.float32)
                path=seqdir/'test'/f'{sid}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                np.savez(path,sample_id=sid,split='test',label=label,actor_id=2,F_sequence=aseq,
                         length=3,padding_mask=np.zeros(3,bool))
                feat=np.concatenate([np.zeros(48),aseq.mean(0)])
                row=dict(sample_id='03-'+sid,trial_id='unused',actor_id=2,emotion_label=CLASSES[label],split='test')
                row.update({f'f{i:03d}':float(x) for i,x in enumerate(feat)});rows.append(row)
            bad=dict(sample_id='03-01-01-01-01-01-01',trial_id='unused',actor_id=1,emotion_label='neutral',split='train')
            bad.update({f'f{i:03d}':'DO_NOT_PARSE' for i in range(176)});rows.append(bad)
            with audio.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            seq=np.stack(visual+[np.full((16,576),np.nan,np.float32)])
            visualfile=root/'visual.npz'
            np.savez(visualfile,sample_id=np.asarray(ids+['01-01-01-01-01-01']),
                     split=np.asarray(['test']*8+['train']),label=np.asarray(labels+[999]),
                     pooled_features=seq.mean(1),sequence_features=seq,probabilities=np.asarray([{}],dtype=object))
            d=load_test_features(audio,seqdir,visualfile,actors,expected_n=8)
            self.assertEqual(len(d['labels']),8)
            self.assertEqual(set(d['splits']),{'test'})
            self.assertTrue(np.isfinite(d['visual_sequence']).all())
            self.assertFalse((seqdir/'train').exists())
            self.assertFalse(any('prob' in k for k in d))


if __name__=='__main__':
    unittest.main()
