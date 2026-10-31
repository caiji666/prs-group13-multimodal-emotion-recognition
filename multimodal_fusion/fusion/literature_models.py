"""Bimodal LMF core and explicitly labelled SupCon adaptations on frozen features."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import head
from .structure_models import StructureModel

METHODS=('projected_concat_control','lmf_rank4','aux_control','aux_supcon_intra','aux_supcon_cross')


def supervised_contrastive(z, labels, modalities, mode='intra', temperature=.1):
    """Average supervised log-probability over positives; excludes self/invalid anchors.

    intra: each modality separately; cross: anchors contrast against the other modality.
    Same-class instances are positives, not false negatives. This is a task adaptation.
    """
    if mode not in ('intra','cross') or temperature<=0:
        raise ValueError('Invalid contrastive configuration')
    allowed=modalities[:,None].eq(modalities[None,:]) if mode=='intra' else modalities[:,None].ne(modalities[None,:])
    allowed=allowed & ~torch.eye(len(z),dtype=torch.bool,device=z.device)
    positive=allowed & labels[:,None].eq(labels[None,:])
    count=positive.sum(1);valid=count>0
    if not valid.any():return z.sum()*0
    similarity=(F.normalize(z,dim=-1) @ F.normalize(z,dim=-1).T)/temperature
    similarity=similarity[valid];allowed=allowed[valid];positive=positive[valid]
    logp=similarity-torch.logsumexp(similarity.masked_fill(~allowed,float('-inf')),dim=1,keepdim=True)
    return -(torch.where(positive,logp,torch.zeros_like(logp)).sum(1)/count[valid]).mean()


class LiteratureModel(nn.Module):
    def __init__(self,method,audio_dim=176,visual_dim=576,audio_seq_dim=128,visual_seq_dim=576,
                 hidden=64,heads=4,dropout=.2,rank=4,temperature=.1,contrastive_weight=.1):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method;self.temperature=temperature;self.contrastive_weight=contrastive_weight
        if method.startswith('aux_'):
            self.core=StructureModel('audio_residual_aux',audio_dim,visual_dim,audio_seq_dim,visual_seq_dim,
                                     hidden=hidden,heads=heads,dropout=dropout)
            # Control uses the same allocated extra head and same main-model/dropout RNG stream.
            rng=torch.get_rng_state()
            self.contrastive_projection=nn.Linear(hidden,hidden)
            torch.set_rng_state(rng)
        else:
            self.audio=nn.Linear(audio_dim,hidden);self.visual=nn.Linear(visual_dim,hidden)
            self.norm=nn.LayerNorm(hidden);self.classifier=head(hidden,hidden,dropout)
            if method=='projected_concat_control':
                self.combine=nn.Linear(2*hidden,hidden)
            else:
                self.audio_factor=nn.Parameter(torch.empty(rank,hidden+1,hidden))
                self.visual_factor=nn.Parameter(torch.empty(rank,hidden+1,hidden))
                nn.init.normal_(self.audio_factor,std=1/math.sqrt(hidden+1))
                nn.init.normal_(self.visual_factor,std=1/math.sqrt(hidden+1))
                self.rank_weights=nn.Parameter(torch.full((rank,),1/rank))
                self.fusion_bias=nn.Parameter(torch.zeros(hidden))

    def forward(self,batch,details=False):
        if self.method.startswith('aux_'):
            logits,aux=self.core(batch,return_aux=True)
            if not details:return logits
            a=torch.tanh(self.core.audio(batch['audio_pooled']))
            v=torch.tanh(self.core.visual(batch['visual_pooled']))
            z=self.contrastive_projection(torch.cat((a,v),0))
            return logits,aux,z
        a=torch.tanh(self.audio(batch['audio_pooled']));v=torch.tanh(self.visual(batch['visual_pooled']))
        if self.method=='projected_concat_control':
            fused=self.combine(torch.cat((a,v),-1))
        else:
            # Appended 1 retains constant and unimodal terms in addition to cross interactions.
            ones=torch.ones(len(a),1,device=a.device,dtype=a.dtype)
            ap=torch.einsum('bd,rdo->bro',torch.cat((a,ones),1),self.audio_factor)
            vp=torch.einsum('bd,rdo->bro',torch.cat((v,ones),1),self.visual_factor)
            fused=((ap*vp)*self.rank_weights[None,:,None]).sum(1)+self.fusion_bias
        logits=self.classifier(self.norm(fused))
        return (logits,(),None) if details else logits
