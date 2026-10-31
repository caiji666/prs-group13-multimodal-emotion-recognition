import copy
import unittest
import torch
from fusion.structure_models import StructureModel, METHODS


class StructureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.config=dict(audio_dim=5,visual_dim=7,audio_seq_dim=3,visual_seq_dim=4,hidden=8,heads=2,dropout=0.)
        self.batch=dict(audio_pooled=torch.randn(2,5),visual_pooled=torch.randn(2,7),
                        audio_sequence=torch.randn(2,6,3),visual_sequence=torch.randn(2,5,4),
                        audio_lengths=torch.tensor([3,6]),visual_lengths=torch.tensor([2,5]))

    def test_all_sequence_variants_ignore_padding_and_extra_padding(self):
        changed={k:v.clone() for k,v in self.batch.items()}
        changed['audio_sequence'][0,3:]=999;changed['visual_sequence'][0,2:]=-999
        for method in METHODS[4:]:
            with self.subTest(method=method):
                model=StructureModel(method,**self.config).eval()
                torch.testing.assert_close(model(self.batch),model(changed))
                extended=copy.copy(changed)
                for mod in ('audio','visual'):
                    x=changed[mod+'_sequence']
                    extended[mod+'_sequence']=torch.cat((x,torch.full((2,3,x.shape[-1]),999.)),1)
                torch.testing.assert_close(model(self.batch),model(extended),atol=1e-6,rtol=1e-5)

    def test_residual_starts_at_exact_frozen_audio_and_keeps_base_eval(self):
        for method in METHODS[:4]:
            model=StructureModel(method,**dict(self.config,dropout=.5)).train()
            self.assertFalse(model.base.training)
            torch.testing.assert_close(model(self.batch),model.base(self.batch),rtol=0,atol=0)
            before=copy.deepcopy(model.base.state_dict())
            logits,aux=model(self.batch,return_aux=True)
            loss=logits.square().mean()+sum(x.square().mean() for x in aux)
            loss.backward()
            self.assertTrue(all(p.grad is None for p in model.base.parameters()))
            self.assertGreater(float(model.correction[-1].weight.grad.abs().sum()),0)
            if 'aux' in method:self.assertGreater(float(model.visual_aux.weight.grad.abs().sum()),0)
            for k,v in model.base.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_uniform_is_invariant_to_valid_visual_order_but_position_is_not(self):
        b={k:v[1:].clone() for k,v in self.batch.items()}
        changed={k:v.clone() for k,v in b.items()}
        changed['visual_sequence']=b['visual_sequence'].flip(1)
        uniform=StructureModel('uniform_attention',**self.config).eval()
        torch.testing.assert_close(uniform(b),uniform(changed),atol=1e-6,rtol=1e-5)
        position=StructureModel('cross_position',**self.config).eval()
        self.assertGreater(float((position(b)-position(changed)).detach().abs().max()),1e-6)

    def test_missing_visual_is_train_only_and_changes_correction(self):
        model=StructureModel('audio_residual_mdrop',**dict(self.config,modality_dropout=1.))
        torch.nn.init.normal_(model.correction[-1].weight)
        changed={k:v.clone() for k,v in self.batch.items()};changed['visual_pooled']+=99
        model.train()
        torch.testing.assert_close(model(self.batch),model(changed))
        model.eval()
        self.assertGreater(float((model(self.batch)-model(changed)).detach().abs().max()),1e-5)


if __name__=='__main__':unittest.main()
