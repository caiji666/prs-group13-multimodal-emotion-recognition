import torch
from research.speech_transfer.models import build

def test_pooled_modes_shapes_gradients_and_reload():
    torch.manual_seed(42)
    for pooling,layer in [('mean',12),('statistics',12),('statistics','mixture')]:
        c=dict(pooling=pooling,layer=layer,dropout=.2,input_dropout=.0)
        m=build(c,torch.zeros(3,1536),torch.ones(3,1536))
        x=torch.randn(4,3,1536)
        assert m.encode(x).shape==(4,128) and m(x).shape==(4,8)
        m(x).square().mean().backward()
        assert m.adapter[0].weight.grad.abs().sum()>0
        if layer=='mixture':assert m.layer_weights.grad.abs().sum()>0
        restored=build(c,m.mean,m.std);restored.load_state_dict(m.state_dict());m.eval();restored.eval()
        torch.testing.assert_close(m(x),restored(x),rtol=0,atol=0)

def test_temporal_mask_ignores_padding_and_exports_sequence():
    torch.manual_seed(42)
    c=dict(pooling='attentive_statistics',layer=9,dropout=.2)
    m=build(c,torch.zeros(768),torch.ones(768)).eval()
    x=torch.randn(3,8,768);lengths=torch.tensor([8,5,3])
    mask=torch.arange(8)[None,:]>=lengths[:,None]
    changed=x.clone();changed[mask]=1000
    torch.testing.assert_close(m(x,lengths),m(changed,lengths))
    extended=torch.cat([x,torch.full((3,4,768),1000.)],1)
    torch.testing.assert_close(m(x,lengths),m(extended,lengths))
    seq,pooled,pad=m.representations(x,lengths)
    assert seq.shape==(3,8,128) and pooled.shape==(3,128)
    assert (seq[pad]==0).all()
