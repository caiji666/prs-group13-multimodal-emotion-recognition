from pathlib import Path
import csv
import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from ravdess_m2.data import parse_filename, read_splits, manifest, extract, collate, SpeechDataset, LABELS
from ravdess_m2.models import AudioEncoder

@pytest.fixture
def audio(tmp_path):
    p=tmp_path/'03-01-06-01-02-01-12.wav'
    sr=22050
    y=np.r_[np.zeros(sr//4), .3*np.sin(2*np.pi*220*np.arange(sr)/sr), np.zeros(sr//4)]
    sf.write(p,y,sr)
    return p

@pytest.fixture
def config(): return yaml.safe_load(Path('configs/default.yaml').read_text())

def test_names_labels():
    r=parse_filename('03-01-06-01-02-01-12.wav')
    assert r['sample_id']=='01-06-01-02-01-12' and r['label']==5
    for e in range(1,9): assert parse_filename(f'03-01-{e:02}-01-01-01-01.wav')['emotion']==LABELS[e-1]
    for name in ['03-02-06-01-02-01-12.wav','03-01-01-02-01-01-01.wav','broken.wav']:
        with pytest.raises(ValueError): parse_filename(name)

def test_splits(tmp_path):
    splits=read_splits('metadata/actor_splits.csv')
    assert {a for a,s in splits.items() if s=='test'}=={2,3,8,17}
    assert {a for a,s in splits.items() if s=='val'}=={9,14,21,22}
    p=tmp_path/'bad.csv'; p.write_text('actor_id,split\n1,train\n1,test\n')
    with pytest.raises(ValueError): read_splits(p)

def test_audio_and_augmentation(audio,config):
    plain=extract(audio,config,'train')
    assert plain.ndim==2 and plain.shape[1]==40 and torch.isfinite(plain).all()
    for split in ('val','test'):
        assert torch.equal(plain,extract(audio,config,split))
        with pytest.raises(ValueError): extract(audio,config,split,True)
        with pytest.raises(ValueError): SpeechDataset([],config,split,True)
    aug=extract(audio,config,'train',True,np.random.default_rng(42))
    assert not torch.equal(plain,aug)
    assert torch.equal(aug,extract(audio,config,'train',True,np.random.default_rng(42)))
    config=dict(config,denoise=True)
    assert torch.isfinite(extract(audio,config,'train')).all()

def test_manifest_duplicates(audio,tmp_path):
    rows=manifest(tmp_path,'metadata/actor_splits.csv',False)
    assert rows[0]['split']=='train'
    with pytest.raises(ValueError): manifest(tmp_path,'metadata/actor_splits.csv')
    (tmp_path/'duplicate').mkdir()
    (tmp_path/'duplicate'/audio.name).write_bytes(audio.read_bytes())
    with pytest.raises(ValueError,match='Duplicate'): manifest(tmp_path,'metadata/actor_splits.csv',False)

@pytest.mark.parametrize('kind',['cnn','bilstm','transformer'])
def test_models_padding_gradients(kind):
    torch.manual_seed(42)
    items=[dict(features=torch.randn(t,40),label=0,sample_id=str(t),split='train') for t in (1,7,12)]
    batch=collate(items)
    assert batch['padding_mask'][0].sum()==11
    model=AudioEncoder(kind).eval()
    result=model(batch['x'],batch['lengths'])
    assert result['logits'].shape==(3,8) and result['pooled'].shape==(3,128)
    assert torch.isfinite(result['sequence']).all()
    assert (result['sequence'][result['padding_mask']]==0).all()
    for index,t in enumerate((1,7,12)):
        single=model(batch['x'][index:index+1,:t],torch.tensor([t]))
        torch.testing.assert_close(single['pooled'][0],result['pooled'][index],atol=2e-5,rtol=2e-5)
    dirty=torch.nn.functional.pad(batch['x'],(0,0,0,5),value=999)
    for i,t in enumerate((1,7,12)): dirty[i,t:]=999
    other=model(dirty,batch['lengths'])
    torch.testing.assert_close(result['pooled'],other['pooled'],atol=2e-5,rtol=2e-5)
    loss=torch.nn.functional.cross_entropy(result['logits'],batch['labels']); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
