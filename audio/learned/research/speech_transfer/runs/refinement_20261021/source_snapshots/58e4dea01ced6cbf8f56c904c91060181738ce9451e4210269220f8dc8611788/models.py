import torch
from torch import nn
from ..models import TemporalAdapter

class ResidualFrame(nn.Module):
    def __init__(self,width):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,width),nn.GELU(),nn.Linear(width,128))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return x+self.net(x)

class ExtendedHead(nn.Sequential):
    def __init__(self,dropout,extra):
        super().__init__(nn.Linear(128,64),nn.ReLU(),nn.Dropout(dropout),nn.Linear(64,8))
        if extra:self.refinement=ResidualFrame(64)
    def forward(self,x):
        if hasattr(self,'refinement'):x=self.refinement(x)
        for key in ('0','1','2','3'):x=self._modules[key](x)
        return x

class RefinementAdapter(TemporalAdapter):
    def __init__(self,config,mean,std):
        super().__init__(config,mean,std)
        self.frame_blocks=nn.ModuleList([ResidualFrame(config.get('frame_width',64)) for _ in range(config.get('frame_depth',0))])
        self.head=ExtendedHead(config['dropout'],config.get('head_residual',False))
        kernel=config.get('time_kernel',0)
        if kernel:
            self.context=nn.Conv1d(128,128,kernel,padding=kernel//2,groups=128)
            nn.init.zeros_(self.context.weight);nn.init.zeros_(self.context.bias)
    def representations(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:]>=lengths[:,None]
        z=(x-self.mean)/self.std
        if self.training and self.config.get('feature_noise',0):z=z+torch.randn_like(z)*self.config['feature_noise']
        sequence=self.projection(z).masked_fill(mask[:,:,None],0)
        for block in self.frame_blocks:sequence=block(sequence).masked_fill(mask[:,:,None],0)
        if hasattr(self,'context'):sequence=(sequence+self.context(sequence.transpose(1,2)).transpose(1,2)).masked_fill(mask[:,:,None],0)
        w=self.attention(sequence).squeeze(-1).masked_fill(mask,float('-inf')).softmax(1)[:,:,None]
        avg=(w*sequence).sum(1);std=((w*(sequence-avg[:,None,:]).square()).sum(1)+1e-5).sqrt()
        return sequence,self.statistics(torch.cat([avg,std],-1)),mask

def build(config,mean,std):return RefinementAdapter(config,mean,std)
