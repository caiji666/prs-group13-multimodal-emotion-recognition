"""Independent WAV prediction/export for the new speech-transfer models."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from transformers import WavLMModel,Wav2Vec2FeatureExtractor
from .prepare import ROOT,LAYERS,REVISION,wave,sha
from .models import build
from ravdess_m2.data import LABELS,parse_filename,read_splits

class SpeechTransferInference:
    def __init__(self,checkpoint):
        torch.set_num_threads(4)
        self.checkpoint=Path(checkpoint)
        self.saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert self.saved['class_names']==LABELS and self.saved['upstream']['revision']==REVISION
        assert sha(ROOT/'assets/wavlm-base-plus/pytorch_model.bin')==self.saved['upstream']['sha256']
        self.config=self.saved['config'];s=self.saved['state_dict']
        self.model=build(self.config,s['mean'],s['std']).eval();self.model.load_state_dict(s,strict=True)
        self.upstream=WavLMModel.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True).eval().requires_grad_(False)
        self.processor=Wav2Vec2FeatureExtractor.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True)
    @torch.inference_mode()
    def predict(self,audio,external=False):
        audio=Path(audio);inputs=self.processor(wave(audio),sampling_rate=16000,return_tensors='pt')
        out=self.upstream(**inputs,output_hidden_states=True)
        h=torch.stack([out.hidden_states[j][0] for j in LAYERS]).half().float()
        if self.config['pooling']=='attentive_statistics':
            x=h[LAYERS.index(self.config['layer'])][None];lengths=torch.tensor([x.shape[1]])
            sequence,pooled,mask=self.model.representations(x,lengths);sequence=sequence[0]
        else:
            statistics=torch.cat([h.mean(1),h.std(1,unbiased=False)],-1)
            pooled=self.model.encode(statistics[None])
            if self.config['layer']=='mixture':sequence=(h*self.model.layer_weights.softmax(0)[:,None,None]).sum(0)
            else:sequence=h[LAYERS.index(self.config['layer'])]
        logits=self.model.head(pooled)[0];pooled=pooled[0];length=len(sequence)
        if external:
            metadata=dict(sample_id='external_'+hashlib.sha256(audio.read_bytes()).hexdigest()[:20],actor_id=-1,label=-1,split='external')
        else:
            metadata=parse_filename(audio);metadata.pop('emotion',None)
            split=ROOT.parents[3]/'multimodal_fusion/configs/actor_splits.csv';metadata['split']=read_splits(split)[metadata['actor_id']]
        return dict(**metadata,sequence=sequence.numpy(),pooled=pooled.numpy(),length=np.int64(length),padding_mask=np.zeros(length,dtype=bool),logits=logits.numpy(),probabilities=logits.softmax(-1).numpy(),class_names=np.array(LABELS),F_sequence=sequence.numpy(),F_audio_learned_feature_only=pooled.numpy(),P_audio_learned_feature_only=logits.softmax(-1).numpy(),sequence_dim=np.int64(sequence.shape[-1]),pooled_dim=np.int64(128),upstream_revision=np.array(REVISION),probability_role=np.array('M2_learned_only_not_M1_final_audio'),sequence_definition=np.array('trained temporal adapter sequence' if self.config['pooling']=='attentive_statistics' else 'frozen WavLM selected/weighted raw hidden states; learned pooled adapter is separate'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--external',action='store_true');a=p.parse_args()
    target=Path(a.output)
    if target.exists():raise FileExistsError(target)
    result=SpeechTransferInference(a.checkpoint).predict(a.audio,a.external)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:np.savez_compressed(f,**result)
    print('Emotion:',LABELS[int(result['probabilities'].argmax())]);print('Eight probabilities:',result['probabilities'].tolist());print('Saved',target,'sequence',result['sequence'].shape,'pooled',result['pooled'].shape)
if __name__=='__main__':main()
