import unittest
import numpy as np
import torch
from fusion.final_models import FinalModel,DeliveredAudioHead
from fusion.final_inputs import view,classes
from fusion import CLASSES


class FinalDeliveryTests(unittest.TestCase):
    def test_delivered_head_undoes_only_m5_standardization(self):
        torch.manual_seed(3);mean=torch.randn(128);std=torch.rand(128)+.1
        stats={'audio_pooled':{'mean':mean.tolist(),'std':std.tolist()}}
        model=DeliveredAudioHead(stats).eval();x=torch.randn(5,128)
        torch.testing.assert_close(model({'audio_pooled':(x-mean)/std}),model.classifier(x),atol=1e-6,rtol=1e-5)
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_final_residual_freezes_original_audio_head_in_train_mode(self):
        stats={'audio_pooled':{'mean':[0.]*128,'std':[1.]*128}}
        config=dict(audio_dim=128,visual_dim=320,audio_seq_dim=128,visual_seq_dim=320,hidden=64,heads=4,dropout=.2)
        m=FinalModel('structure','audio_residual',config,stats).train()
        self.assertFalse(m.inner.base.training)
        b={'audio_pooled':torch.randn(2,128),'visual_pooled':torch.randn(2,320)}
        torch.testing.assert_close(m(b),m.inner.base(b),rtol=0,atol=0)
        m(b).square().sum().backward();self.assertTrue(all(p.grad is None for p in m.inner.base.parameters()))

    def test_representation_view_does_not_confuse_pooled_with_sequence_mean(self):
        raw={k:np.zeros(2) for k in ('sample_ids','labels','actors','splits','audio_sequence','audio_lengths','visual_lengths')}
        raw.update(audio_pooled=np.full((2,128),7),audio_handcrafted_pooled=np.ones((2,48)),
                   audio_combined_pooled=np.ones((2,176)),visual_hc_pooled=np.ones((2,320)),visual_hc_sequence=np.ones((2,16,320)))
        d=view(raw);self.assertEqual(d['audio_pooled'].shape,(2,128));self.assertEqual(d['audio_pooled'][0,0],7)
        self.assertEqual(view(raw,audio='handcrafted')['audio_pooled'].shape,(2,48))
        self.assertEqual(classes(['neutral','calm','happy','sad','angry','fearful','disgust','surprised']),CLASSES)


if __name__=='__main__':unittest.main()
