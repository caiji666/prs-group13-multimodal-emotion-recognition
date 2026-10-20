import unittest
import torch
from fusion.models import FusionModel
from fusion.study_diagnostics import attention_probe


class StudyControlsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        self.config=dict(audio_dim=5,visual_dim=7,audio_seq_dim=3,visual_seq_dim=4,hidden=8,heads=2,dropout=0.)
        self.batch={'audio_pooled':torch.randn(2,5),'visual_pooled':torch.randn(2,7),
                    'audio_sequence':torch.randn(2,6,3),'visual_sequence':torch.randn(2,4,4),
                    'audio_lengths':torch.tensor([3,6]),'visual_lengths':torch.tensor([2,4])}

    def test_single_modality_controls_ignore_other_modality(self):
        for method in ('audio_only','visual_only','audio_sequence_mean'):
            model=FusionModel(method,**self.config).eval()
            other='audio' if method=='visual_only' else 'visual'
            changed={k:v.clone() for k,v in self.batch.items()}
            for k in (other+'_pooled',other+'_sequence'):changed[k]+=1000
            torch.testing.assert_close(model(self.batch),model(changed))

    def test_sequence_controls_exclude_padding(self):
        changed={k:v.clone() for k,v in self.batch.items()}
        changed['audio_sequence'][0,3:]=1000
        changed['visual_sequence'][0,2:]=-1000
        for method in ('audio_sequence_mean','sequence_mean_concat'):
            model=FusionModel(method,**self.config).eval()
            torch.testing.assert_close(model(self.batch),model(changed))

    def test_attention_probe_preserves_forward_and_masks(self):
        model=FusionModel('cross_attention',**self.config).eval()
        logits,saliency,weights=attention_probe(model,self.batch)
        torch.testing.assert_close(logits,model(self.batch),rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(saliency.sum(-1),torch.ones(2))
        self.assertEqual(float(saliency[0,2:].abs().sum()),0.)
        modified={k:v.clone() for k,v in self.batch.items()}
        modified['visual_sequence'][0,2:]=1000
        uniform=attention_probe(model,self.batch,True)[0]
        torch.testing.assert_close(uniform,attention_probe(model,modified,True)[0])


if __name__=='__main__':
    unittest.main()
