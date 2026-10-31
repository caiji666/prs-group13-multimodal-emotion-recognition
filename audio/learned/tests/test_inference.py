import hashlib
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from ravdess_m2.data import LABELS,extract
from ravdess_m2.inference import AudioInference
from ravdess_m2.models import AudioEncoder


def test_new_audio_frozen_inference_and_ravdess_metadata(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(42)
    preprocessing=yaml.safe_load(Path('configs/default.yaml').read_text())
    model=AudioEncoder('cnn').eval()
    model.feature_mean.fill_(1); model.feature_std.fill_(2)
    checkpoint=tmp_path/'best_audio_encoder.pt'
    torch.save(dict(encoder='cnn',class_names=LABELS,preprocessing=preprocessing,model_state_dict=model.state_dict(),
                    provenance=dict(diagnostic=False,split_sha256=hashlib.sha256(Path('metadata/actor_splits.csv').read_bytes()).hexdigest())),checkpoint)
    sr=16000
    wav=tmp_path/'new_audio.wav'
    sf.write(wav,.3*np.sin(2*np.pi*220*np.arange(sr)/sr),sr)
    inference=AudioInference(checkpoint)
    result=inference.predict(wav)
    assert str(result['split'])=='external' and int(result['label'])==-1 and int(result['actor_id'])==-1
    assert str(result['sample_id']).startswith('external-')
    assert result['pooled'].shape==(128,) and result['probabilities'].shape==(8,)
    np.testing.assert_array_equal(result['F_sequence'],result['sequence'])
    np.testing.assert_array_equal(result['F_audio_learned_feature_only'],result['pooled'])
    np.testing.assert_array_equal(result['P_audio_learned_feature_only'],result['probabilities'])
    assert np.isclose(result['probabilities'].sum(),1)
    assert str(result['probability_role'])=='M2_learned_only_not_final_P_audio'
    x=extract(wav,preprocessing,'train',augment=False)
    with torch.inference_mode(): expected=model(x[None],torch.tensor([len(x)]))
    np.testing.assert_allclose(result['pooled'],expected['pooled'][0].numpy(),atol=1e-6)
    again=inference.predict(wav)
    np.testing.assert_array_equal(again['pooled'],result['pooled'])
    rav=tmp_path/'03-01-06-01-02-01-12.wav'; rav.write_bytes(wav.read_bytes())
    with pytest.raises(ValueError,match='requires --splits'): inference.predict(rav)
    labeled=inference.predict(rav,'metadata/actor_splits.csv')
    assert str(labeled['sample_id'])=='01-06-01-02-01-12'
    assert str(labeled['split'])=='train' and int(labeled['label'])==5
    external=inference.predict(rav,external=True)
    assert str(external['split'])=='external' and int(external['label'])==-1
    wrong=tmp_path/'wrong.csv'; wrong.write_text('actor_id,split\n1,test\n')
    with pytest.raises(ValueError,match='differs'): inference.predict(rav,wrong)
