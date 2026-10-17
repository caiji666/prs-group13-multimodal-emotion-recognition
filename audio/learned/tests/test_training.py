"""Tiny synthetic tests only; no formal RAVDESS training."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader
import ravdess_m2.train as training
from ravdess_m2.data import LABELS, collate
from ravdess_m2.models import AudioEncoder

@pytest.fixture
def config():
    torch.set_num_threads(1)
    c=yaml.safe_load(Path('configs/train.yaml').read_text())
    c.update(epochs=2,batch_size=4,augment=False)
    return c

@pytest.fixture
def tiny_data():
    rng=torch.Generator().manual_seed(123)
    rows=[]; clean={}
    for split in ('train','val'):
        for i in range(8):
            sid=f'{split}-{i}'
            rows.append(dict(sample_id=sid,split=split,label=i,path='not-a-real-file'))
            clean[sid]=torch.randn(5+i,40,generator=rng)
    return rows,clean

def test_metric_definition():
    perfect=training.metrics(list(range(8)),list(range(8)))
    assert perfect['macro_f1']==perfect['accuracy']==1
    wrong=training.metrics([0,1],[1,1])
    assert wrong['accuracy']==.5
    assert wrong['macro_precision']==pytest.approx(.5/8)
    assert wrong['macro_recall']==pytest.approx(1/8)
    assert wrong['macro_f1']==pytest.approx((2/3)/8)
    assert wrong['confusion_matrix'][0][1]==1
    with pytest.raises(ValueError): training.metrics([],[])

def test_train_only_normalization_and_augmentation(tiny_data,monkeypatch):
    rows,clean=tiny_data
    train=[clean[r['sample_id']] for r in rows if r['split']=='train']
    mean,std=training.fit_standardizer(train)
    full=torch.cat(train)
    torch.testing.assert_close(mean,full.mean(0))
    torch.testing.assert_close(std,full.std(0,unbiased=False))
    with pytest.raises(ValueError): training.TrainingDataset(rows,{},clean,'test',False,42)
    with pytest.raises(ValueError): training.TrainingDataset(rows,{},clean,'val',True,42)
    calls=[]
    def fake_extract(path,config,split,augment,rng):
        calls.append((split,augment))
        return torch.from_numpy(rng.normal(size=(5,40)).astype('float32'))
    monkeypatch.setattr(training,'extract',fake_extract)
    a=training.TrainingDataset(rows,{},clean,'train',True,42)
    b=training.TrainingDataset(rows,{},clean,'train',True,42)
    a.epoch=b.epoch=1
    torch.testing.assert_close(a[0]['features'],b[0]['features'])
    b.epoch=2
    assert not torch.equal(a[0]['features'],b[0]['features'])
    assert all(split=='train' and augment for split,augment in calls)
    val=training.TrainingDataset(rows,{},clean,'val',False,42)
    torch.testing.assert_close(val[0]['features'],clean['val-0'])

def test_three_independent_training_checkpoint_roundtrips(tmp_path,config,tiny_data):
    rows,clean=tiny_data
    mean,std=training.fit_standardizer([clean[r['sample_id']] for r in rows if r['split']=='train'])
    train=training.TrainingDataset(rows,{},clean,'train',False,42)
    val=training.TrainingDataset(rows,{},clean,'val',False,42)
    provenance=dict(protocol_sha256='test-protocol',diagnostic=True,split_sha256='test-split')
    results=[]
    for kind in training.KINDS:
        result=training.train_one(kind,train,val,config,{'n_mfcc':40},mean,std,tmp_path,provenance)
        results.append(result)
        best=torch.load(tmp_path/kind/'best.pt',weights_only=True)
        last=torch.load(tmp_path/kind/'last.pt',weights_only=True)
        assert last['epoch']==2 and 'optimizer_state_dict' in last and 'scheduler_state_dict' in last
        assert best['encoder']==kind and best['class_names']==LABELS
        history=json.loads((tmp_path/kind/'history.json').read_text())
        assert best['best_val_macro_f1']==max(h['val']['macro_f1'] for h in history)
        restored=AudioEncoder(kind); restored.load_state_dict(best['model_state_dict']); restored.eval()
        torch.testing.assert_close(restored.feature_mean,mean)
        torch.testing.assert_close(restored.feature_std,std)
        recomputed,_=training.run_epoch(restored,DataLoader(val,batch_size=4,collate_fn=collate),'cpu')
        assert recomputed['macro_f1']==result['best_val_macro_f1']
        training.seed_all(42)
        initial=AudioEncoder(kind)
        assert not torch.equal(initial.head[0].weight,restored.head[0].weight)
        assert (tmp_path/kind/'best_val_confusion_matrix.csv').exists()
        assert len((tmp_path/kind/'best_val_predictions.csv').read_text().splitlines())==9
    report=training.compare(results,tmp_path,diagnostic=True)
    assert report['selected_encoder'] is None and report['test_evaluated'] is False
    report=training.compare(results,tmp_path,diagnostic=False)
    assert report['selected_encoder']==max(results,key=lambda r:r['best_val_macro_f1'])['encoder']
    with pytest.raises(ValueError): training.compare(results[:2],tmp_path)
    results[0]['protocol_sha256']='different'
    with pytest.raises(ValueError): training.compare(results,tmp_path)

def test_early_stop_and_best_is_not_last(tmp_path,config,tiny_data,monkeypatch):
    config.update(epochs=10,patience=2)
    rows,clean=tiny_data
    train=training.TrainingDataset(rows,{},clean,'train',False,42)
    val=training.TrainingDataset(rows,{},clean,'val',False,42)
    values=iter([.4,.6,.5,.5])
    def fake_epoch(model,loader,device,optimizer=None,grad_clip=1.):
        m=training.metrics(list(range(8)),list(range(8))); m['loss']=1.
        if optimizer is None: m['macro_f1']=next(values)
        return m,[dict(sample_id='fake',split='val',label=0,prediction=0)]
    monkeypatch.setattr(training,'run_epoch',fake_epoch)
    result=training.train_one('cnn',train,val,config,{'n_mfcc':40},torch.zeros(40),torch.ones(40),tmp_path,dict(protocol_sha256='x',diagnostic=True))
    assert result['best_epoch']==2 and result['epochs_completed']==4
    assert torch.load(tmp_path/'cnn/best.pt',weights_only=True)['epoch']==2
    assert torch.load(tmp_path/'cnn/last.pt',weights_only=True)['epoch']==4

def test_config_and_no_test_loader(config,tiny_data):
    training.validate_config(config)
    with pytest.raises(ValueError): training.validate_config(dict(config,epochs=0))
    with pytest.raises(ValueError): training.validate_config(dict(config,num_workers=2))
    rows,clean=tiny_data
    ds=training.TrainingDataset(rows,{},clean,'val',False,42)
    ds.rows[0]=dict(ds.rows[0],split='test')
    with pytest.raises(ValueError,match='loader contains'):
        training.run_epoch(AudioEncoder('cnn'),DataLoader(ds,batch_size=4,collate_fn=collate),'cpu')

def test_entrypoint_excludes_test_and_refuses_overwrite(tmp_path,tiny_data,monkeypatch):
    import sys
    rows,clean=tiny_data
    rows=rows+[dict(sample_id='forbidden',split='test',label=0,path='forbidden.wav')]
    monkeypatch.setattr(training,'manifest',lambda *args,**kwargs:rows)
    called=[]
    def fake_extract(path,preprocessing,split):
        assert split in ('train','val') and path!='forbidden.wav'
        called.append(split)
        return torch.ones(4,40)
    monkeypatch.setattr(training,'extract',fake_extract)
    def fake_train(kind,train,val,config,preprocessing,mean,std,run_dir,provenance):
        assert set(r['split'] for r in train.rows)=={'train'}
        assert set(r['split'] for r in val.rows)=={'val'}
        return dict(encoder=kind,best_epoch=1,best_val_macro_f1=.5,checkpoint=f'{kind}/best.pt',
                    protocol_sha256=provenance['protocol_sha256'],diagnostic=False,
                    validation_metrics=dict(macro_f1=.5,accuracy=.5,macro_precision=.5,macro_recall=.5))
    monkeypatch.setattr(training,'train_one',fake_train)
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--output',str(out),'--epochs','1'])
    training.main()
    assert called.count('train')==8 and called.count('val')==8
    protocol=json.loads((out/'protocol.json').read_text())
    assert all(r['split']!='test' for r in protocol['samples'])
    report=json.loads((out/'comparison.json').read_text())
    assert report['selected_encoder']=='cnn' and report['test_evaluated'] is False
    before=(out/'comparison.json').read_bytes()
    with pytest.raises(FileExistsError): training.main()
    assert (out/'comparison.json').read_bytes()==before


def test_export_uses_checkpoint_normalization(tmp_path,monkeypatch):
    import csv
    import hashlib
    import sys
    import ravdess_m2.dataset_tools as cli
    pre=yaml.safe_load(Path('configs/default.yaml').read_text())
    model=AudioEncoder('cnn').eval()
    model.feature_mean.fill_(2); model.feature_std.fill_(3)
    checkpoint=tmp_path/'trained_for_test.pt'
    torch.save(dict(encoder='cnn',model_state_dict=model.state_dict(),preprocessing=pre,
                    class_names=LABELS,provenance=dict(diagnostic=False,split_sha256=hashlib.sha256(Path('metadata/actor_splits.csv').read_bytes()).hexdigest())),checkpoint)
    row=dict(sample_id='01-06-01-02-01-12',split='train',label=5,actor_id=12,path='synthetic')
    monkeypatch.setattr(cli,'manifest',lambda *args,**kwargs:[row])
    features=torch.randn(9,40)
    monkeypatch.setattr(cli,'extract',lambda *args,**kwargs:features)
    out=tmp_path/'export'
    monkeypatch.setattr(sys,'argv',['cli','export','--data','not-read','--checkpoint',str(checkpoint),'--output',str(out)])
    cli.main()
    z=np.load(out/'cnn/train/01-06-01-02-01-12.npz')
    with torch.no_grad(): expected=model(features[None],torch.tensor([9]))
    np.testing.assert_allclose(z['pooled'],expected['pooled'][0].numpy(),rtol=1e-5)
    assert z['logits'].shape==(8,) and np.isclose(z['probabilities'].sum(),1)
    saved=torch.load(checkpoint,weights_only=True)
    saved['provenance']['diagnostic']=True
    torch.save(saved,checkpoint)
    with pytest.raises(SystemExit): cli.main()


def test_warm_start_preserves_lineage_and_rejects_incompatible_inputs(tmp_path,config,tiny_data):
    rows,clean=tiny_data
    train=training.TrainingDataset(rows,{},clean,'train',False,42)
    val=training.TrainingDataset(rows,{},clean,'val',False,42)
    mean,std=training.fit_standardizer([clean[r['sample_id']] for r in rows if r['split']=='train'])
    parent=tmp_path/'parent'; parent.mkdir()
    provenance=dict(protocol_sha256='synthetic',diagnostic=False,split_sha256='same-split')
    config=dict(config,epochs=1)
    training.train_one('cnn',train,val,config,{'n_mfcc':40},mean,std,parent,provenance)
    (parent/'status.json').write_text('{"status":"completed"}')
    child=tmp_path/'child'; child.mkdir()
    tuned=dict(config,warm_start={'cnn':str(parent)},class_balance=True,label_smoothing=.05,head_dropout=.3,learning_rate=.0001)
    result=training.train_one('cnn',train,val,tuned,{'n_mfcc':40},mean,std,child,provenance)
    assert result['initialization']['parent_epoch']==1
    saved=torch.load(child/'cnn/best.pt',weights_only=True)
    assert saved['initialization']['sha256']==result['initialization']['sha256']
    assert saved['train_config']['class_balance'] is True
    with pytest.raises(ValueError,match='Incompatible'):
        training.train_one('cnn',train,val,tuned,{'n_mfcc':39},mean,std,child,provenance)


def test_spawned_candidate_matches_serial_training(tmp_path,config,tiny_data):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    rows,clean=tiny_data
    mean,std=training.fit_standardizer([clean[r['sample_id']] for r in rows if r['split']=='train'])
    provenance=dict(protocol_sha256='test',diagnostic=True,split_sha256='test')
    serial=tmp_path/'serial'; serial.mkdir()
    parallel=tmp_path/'parallel'; parallel.mkdir()
    train=training.TrainingDataset(rows,{},clean,'train',False,42)
    val=training.TrainingDataset(rows,{},clean,'val',False,42)
    a=training.train_one('cnn',train,val,config,{'n_mfcc':40},mean,std,serial,provenance)
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        b=pool.submit(training.candidate_worker,'cnn',rows,{k:v.numpy() for k,v in clean.items()},config,{'n_mfcc':40},mean.numpy(),std.numpy(),str(parallel),provenance,None).result()
    assert a['best_val_macro_f1']==b['best_val_macro_f1']
    x=torch.load(serial/'cnn/best.pt',weights_only=True)['model_state_dict']
    y=torch.load(parallel/'cnn/best.pt',weights_only=True)['model_state_dict']
    for key in x: torch.testing.assert_close(x[key],y[key],rtol=0,atol=0)


def test_tempered_class_weights_use_training_counts_only():
    labels=[0]+[i for i in range(1,8) for _ in range(2)]
    full=training.class_weight_vector(labels,1.)
    tempered=training.class_weight_vector(labels,.5)
    assert (full[0]/full[1]).item()==2.
    assert (tempered[0]/tempered[1]).item()==pytest.approx(2**.5)
    with pytest.raises(ValueError): training.class_weight_vector([0,1],.5)
