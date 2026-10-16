"""Copy into a teammate's frozen encoder feature-export script."""
from pathlib import Path
import numpy as np
from fusion import CLASSES
from fusion.data import trial_id


def export_features(path, filenames, pooled, probs, sequence=None, lengths=None):
    """Export detached NumPy arrays. Use encoder.eval() + torch.inference_mode().

    Train/select encoders with the agreed actors.json before exporting all splits.
    probs must be softmax probabilities; reorder SVM predict_proba columns.
    Dimensions may differ between audio/visual and pooled/sequence representations.
    """
    ids = np.asarray([trial_id(f) for f in filenames])
    data = dict(sample_ids=ids, class_names=np.asarray(CLASSES),
                labels=np.asarray([int(s.split('-')[1])-1 for s in ids],dtype=np.int64),
                actors=np.asarray([int(s.split('-')[-1]) for s in ids],dtype=np.int64),
                pooled=np.asarray(pooled,dtype=np.float32), probs=np.asarray(probs,dtype=np.float32))
    if sequence is not None:
        if lengths is None:
            raise ValueError('lengths is required with sequence')
        data.update(sequence=np.asarray(sequence,dtype=np.float32), lengths=np.asarray(lengths,dtype=np.int64))
    elif lengths is not None:
        raise ValueError('sequence is required with lengths')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**data)
