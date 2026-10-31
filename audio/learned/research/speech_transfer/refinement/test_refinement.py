import torch
from research.speech_transfer.models import TemporalAdapter
from research.speech_transfer.refinement.models import build

def test_extensions_preserve_parent_initial_function_and_padding():
    torch.manual_seed(42);torch.set_num_threads(4)
    c=dict(dropout=.3,frame_depth=2,frame_width=64,time_kernel=3,head_residual=True)
    parent=TemporalAdapter(c,torch.zeros(768),torch.ones(768)).eval()
    model=build(c,parent.mean,parent.std).eval();model.load_state_dict(parent.state_dict(),strict=False)
    x=torch.randn(3,10,768);lengths=torch.tensor([10,6,3])
    torch.testing.assert_close(model(x,lengths),parent(x,lengths),rtol=0,atol=0)
    padded=torch.cat([x,torch.full((3,4,768),999.)],1)
    torch.testing.assert_close(model(padded,lengths),model(x,lengths))
    seq,pool,mask=model.representations(x,lengths)
    assert seq.shape==(3,10,128) and pool.shape==(3,128) and not seq[mask].any()
    restored=build(c,parent.mean,parent.std).eval();restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x,lengths),model(x,lengths),rtol=0,atol=0)
    model.train();model(x,lengths).square().mean().backward()
    assert model.context.weight.grad is not None and model.frame_blocks[0].net[-1].weight.grad is not None
