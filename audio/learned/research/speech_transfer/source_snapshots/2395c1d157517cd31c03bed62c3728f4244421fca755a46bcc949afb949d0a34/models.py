"""New downstream adapters for frozen WavLM; legacy encoders are untouched."""
import torch
from torch import nn

class PooledAdapter(nn.Module):
    def __init__(self,config,mean,std):
        super().__init__();self.config=config
        self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.layer_weights=nn.Parameter(torch.zeros(3)) if config.get('layer')=='mixture' else None
        dim=768 if config['pooling']=='mean' else 1536
        self.input_dropout=nn.Dropout(config.get('input_dropout',0.))
        self.adapter=nn.Sequential(nn.Linear(dim,128),nn.LayerNorm(128),nn.GELU())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(64,8))
    def encode(self,features,lengths=None):
        x=(features-self.mean)/self.std
        if self.config['pooling']=='mean':x=x[:,:,:768]
        if self.layer_weights is None:x=x[:,[6,9,12].index(self.config['layer'])]
        else:x=(x*self.layer_weights.softmax(0)[None,:,None]).sum(1)
        return self.adapter(self.input_dropout(x))
    def forward(self,features,lengths=None):return self.head(self.encode(features,lengths))

class TemporalAdapter(nn.Module):
    def __init__(self,config,mean,std):
        super().__init__();self.config=config
        self.register_buffer('mean',mean.clone());self.register_buffer('std',std.clone())
        self.projection=nn.Sequential(nn.Linear(768,128),nn.LayerNorm(128),nn.GELU())
        self.attention=nn.Sequential(nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1))
        self.statistics=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU())
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(config['dropout']),nn.Linear(64,8))
    def representations(self,x,lengths):
        mask=torch.arange(x.shape[1],device=x.device)[None,:]>=lengths[:,None]
        sequence=self.projection((x-self.mean)/self.std).masked_fill(mask[:,:,None],0)
        scores=self.attention(sequence).squeeze(-1).masked_fill(mask,float('-inf'))
        w=scores.softmax(1)[:,:,None]
        avg=(w*sequence).sum(1)
        std=((w*(sequence-avg[:,None,:]).square()).sum(1)+1e-5).sqrt()
        return sequence,self.statistics(torch.cat([avg,std],-1)),mask
    def encode(self,x,lengths):return self.representations(x,lengths)[1]
    def forward(self,x,lengths):return self.head(self.encode(x,lengths))

def build(config,mean,std):
    cls=TemporalAdapter if config['pooling']=='attentive_statistics' else PooledAdapter
    return cls(config,mean,std)
