"""Frozen M2 feature extraction for a new WAV; probabilities are learned-only."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch

from .data import LABELS, extract, parse_filename, read_splits
from .models import AudioEncoder


class AudioInference:
    def __init__(self, checkpoint):
        self.checkpoint_path=Path(checkpoint)
        saved=torch.load(self.checkpoint_path,map_location='cpu',weights_only=True)
        if saved.get('class_names')!=LABELS or saved.get('provenance',{}).get('diagnostic',False):
            raise ValueError('Require a formal checkpoint with the shared class order')
        self.config=saved['preprocessing']
        self.split_sha256=saved['provenance']['split_sha256']
        self.encoder=saved['encoder']
        self.model=AudioEncoder(self.encoder,input_dim=self.config['n_mfcc']).eval()
        self.model.load_state_dict(saved['model_state_dict'],strict=True)
        self.model.requires_grad_(False)
        if not torch.isfinite(self.model.feature_mean).all() or not torch.isfinite(self.model.feature_std).all() or not (self.model.feature_std>0).all():
            raise ValueError('Invalid checkpoint normalization')

    def predict(self, audio, splits=None, external=False):
        audio=Path(audio)
        if not audio.is_file(): raise FileNotFoundError(audio)
        try:
            row=None if external else parse_filename(audio)
        except ValueError:
            row=None
        if row is None:
            row=dict(sample_id='external-'+hashlib.sha256(audio.read_bytes()).hexdigest()[:16],split='external',label=-1,actor_id=-1)
        else:
            splits=Path(splits) if splits else self.checkpoint_path.parent/'actor_splits.csv'
            if not splits.is_file():
                raise ValueError('RAVDESS audio requires --splits; use --external for an unrelated recording')
            if hashlib.sha256(splits.read_bytes()).hexdigest()!=self.split_sha256:
                raise ValueError('Actor split differs from training')
            row['split']=read_splits(splits)[row['actor_id']]
        # external is metadata, not a dataset split used for fitting or augmentation.
        x=extract(audio,self.config,'train' if row['split']=='external' else row['split'],augment=False)
        with torch.inference_mode():
            result=self.model(x[None],torch.tensor([len(x)]))
        output=dict(sample_id=np.array(row['sample_id']),split=np.array(row['split']),label=np.int64(row['label']),
                    actor_id=np.int64(row['actor_id']),sequence=result['sequence'][0].numpy(),pooled=result['pooled'][0].numpy(),
                    length=np.int64(result['lengths'][0]),padding_mask=result['padding_mask'][0].numpy(),
                    logits=result['logits'][0].numpy(),probabilities=result['logits'].softmax(-1)[0].numpy(),
                    class_names=np.array(LABELS),probability_role=np.array('M2_learned_only_not_final_P_audio'))
        output['F_sequence']=output['sequence']
        output['F_audio_learned_feature_only']=output['pooled']
        output['P_audio_learned_feature_only']=output['probabilities']
        return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='best_audio_encoder.pt')
    parser.add_argument('--audio',required=True)
    parser.add_argument('--output',required=True,help='New NPZ path; existing files are rejected')
    parser.add_argument('--splits')
    parser.add_argument('--external',action='store_true',help='Treat recording as unlabeled external input regardless of filename')
    args=parser.parse_args()
    output=Path(args.output)
    if output.suffix!='.npz': parser.error('--output must end in .npz')
    if output.exists(): raise FileExistsError(output)
    torch.set_num_threads(2)
    result=AudioInference(args.checkpoint).predict(args.audio,args.splits,args.external)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f: np.savez_compressed(f,**result)
    print(f"Saved {output}: sequence={result['sequence'].shape}, pooled={result['pooled'].shape}. Probabilities are M2 learned-only, not final P_audio.")

if __name__=='__main__': main()
