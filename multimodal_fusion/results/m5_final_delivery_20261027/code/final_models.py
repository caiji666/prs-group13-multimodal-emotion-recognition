"""Final-input model factory, preserving the delivered WavLM classifier as residual base."""
import torch
from torch import nn
from .models import FusionModel,head
from .structure_models import StructureModel
from .literature_models import LiteratureModel


class DeliveredAudioHead(nn.Module):
    def __init__(self,stats):
        super().__init__()
        self.register_buffer('mean',torch.tensor(stats['audio_pooled']['mean'],dtype=torch.float32))
        self.register_buffer('std',torch.tensor(stats['audio_pooled']['std'],dtype=torch.float32))
        assert len(self.mean)==128
        self.classifier=head(128,64,.3)
        self.requires_grad_(False)

    def forward(self,batch):
        return self.classifier(batch['audio_pooled']*self.std+self.mean)


class FinalModel(nn.Module):
    def __init__(self,family,method,config,stats,source_state=None):
        super().__init__();self.family=family;self.method=method
        cls={'base':FusionModel,'structure':StructureModel,'literature':LiteratureModel}[family]
        self.inner=cls(method,**config)
        target=self.inner.core if family=='literature' and method.startswith('aux_') else self.inner
        self.has_delivered_base=hasattr(target,'base')
        if self.has_delivered_base:
            target.base=DeliveredAudioHead(stats)
            if source_state is not None:
                target.base.classifier.load_state_dict({k:source_state['head.'+k] for k in target.base.classifier.state_dict()})
            target.base.eval()

    def forward(self,batch):return self.inner(batch)

    def training_outputs(self,batch):
        if self.family=='literature':return self.inner(batch,details=True)
        if self.family=='structure':
            logits,aux=self.inner(batch,return_aux=True);return logits,aux,None
        return self.inner(batch),(),None
