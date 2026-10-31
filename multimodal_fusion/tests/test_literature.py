import csv
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fusion.models import FusionModel
from fusion.literature_models import LiteratureModel,supervised_contrastive
from fusion.literature_study import gate_logits,save_predictions


class LiteratureTests(unittest.TestCase):
    def test_supcon_excludes_self_and_uses_same_class_positives(self):
        z=torch.ones(4,3,requires_grad=True);y=torch.zeros(4,dtype=torch.long);mod=torch.tensor([0,0,1,1])
        self.assertAlmostEqual(float(supervised_contrastive(z,y,mod,'intra').detach()),0.,places=6)
        cross=supervised_contrastive(z,y,mod,'cross')
        self.assertAlmostEqual(float(cross.detach()),math.log(2),places=5)
        cross.backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_supcon_no_positive_is_zero_and_has_finite_gradients(self):
        z=torch.randn(4,5,requires_grad=True);loss=supervised_contrastive(z,torch.arange(4),torch.zeros(4,dtype=torch.long))
        self.assertEqual(float(loss.detach()),0.);loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertEqual(float(z.grad.abs().sum()),0.)

    def test_lmf_factorized_output_equals_explicit_dense_tensor(self):
        torch.manual_seed(5)
        m=LiteratureModel('lmf_rank4',audio_dim=3,visual_dim=5,hidden=4,heads=2,rank=2,dropout=0.).eval()
        b=dict(audio_pooled=torch.randn(3,3),visual_pooled=torch.randn(3,5))
        a=torch.cat((torch.tanh(m.audio(b['audio_pooled'])),torch.ones(3,1)),1)
        v=torch.cat((torch.tanh(m.visual(b['visual_pooled'])),torch.ones(3,1)),1)
        weight=torch.einsum('rdo,reo,r->deo',m.audio_factor,m.visual_factor,m.rank_weights)
        dense=torch.einsum('bd,be,deo->bo',a,v,weight)+m.fusion_bias
        torch.testing.assert_close(m(b),m.classifier(m.norm(dense)),atol=1e-6,rtol=1e-5)
        m(b).square().sum().backward()
        self.assertGreater(float(m.audio_factor.grad.abs().sum()),0.)
        self.assertGreater(float(m.visual_factor.grad.abs().sum()),0.)

    def test_gate_override_reproduces_forward_and_does_not_mutate_inputs(self):
        torch.manual_seed(6)
        m=FusionModel('gated',3,5,hidden=8,heads=2,dropout=0.).eval()
        b=dict(audio_pooled=torch.randn(4,3),visual_pooled=torch.randn(4,5));before={k:v.clone() for k,v in b.items()}
        g=torch.sigmoid(m.gate(torch.cat((b['audio_pooled'],b['visual_pooled']),-1)))
        torch.testing.assert_close(m(b),gate_logits(m,b,g))
        changed=gate_logits(m,b,g.flip(0))
        self.assertGreater(float((changed-m(b)).detach().abs().max()),1e-5)
        for k,v in b.items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_svm_export_leaves_probability_columns_empty(self):
        with tempfile.TemporaryDirectory() as t:
            folder=Path(t);raw=dict(sample_ids=np.array(['01-01-01-01-01-09']),actors=np.array([9]),labels=np.array([0]))
            save_predictions(folder,raw,np.array([0]),'run',np.array([1]),scores=np.arange(8)[None,:])
            with (folder/'predictions_val.csv').open() as f:row=next(csv.DictReader(f))
            self.assertEqual(row['y_pred'],'1');self.assertEqual(row['p_calm'],'')
            self.assertTrue((folder/'decision_scores_val.csv').exists())


if __name__=='__main__':unittest.main()
