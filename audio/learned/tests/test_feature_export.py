"""Feature handoff round trips and leakage/provenance guards."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import ravdess_m2.feature_export as feature_export
from ravdess_m2.data import LABELS
from ravdess_m2.feature_io import load_pooled,align_pooled,load_sequence,pad_sequences,validate_bundle
from ravdess_m2.models import AudioEncoder


def write(path,value): path.write_text(json.dumps(value))

@pytest.fixture
def run(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(42)
    run=tmp_path/'run'; (run/'transformer').mkdir(parents=True)
    write(run/'status.json',dict(status='completed',diagnostic=False))
    selection=dict(selection_metric='validation_macro_f1',test_evaluated=False,diagnostic=False,
                   selected_encoder='transformer',selected_checkpoint='transformer/best.pt',
                   comparison=[dict(encoder=k,macro_f1=v,best_epoch=2,checkpoint=f'{k}/best.pt')
                               for k,v in [('cnn',.3),('bilstm',.4),('transformer',.5)]])
    write(run/'comparison.json',selection); (run/'comparison.csv').write_text('fixture\n')
    protocol=dict(protocol_sha256='synthetic',source_sha256={n:feature_export.sha256(Path(feature_export.__file__).with_name(n)) for n in ('models.py','data.py')})
    write(run/'protocol.json',protocol)
    model=AudioEncoder('transformer').eval()
    model.feature_mean.fill_(2); model.feature_std.fill_(3)
    checkpoint=dict(encoder='transformer',epoch=2,model_state_dict=model.state_dict(),preprocessing={'n_mfcc':40},
                    class_names=LABELS,validation_metrics={'macro_f1':.5},
                    provenance=dict(diagnostic=False,split_sha256=feature_export.sha256('metadata/actor_splits.csv'),protocol_sha256='synthetic'))
    torch.save(checkpoint,run/'transformer/best.pt')
    return run,model

@pytest.fixture
def bundle(tmp_path,run,monkeypatch):
    run,model=run
    rows=[]
    for actor,split in [(1,'train'),(4,'train'),(9,'val'),(2,'test')]:
        sid=f'01-03-01-01-01-{actor:02d}'
        rows.append(dict(sample_id=sid,split=split,label=2,actor_id=actor,emotion='happy',path=str(actor)))
    monkeypatch.setattr(feature_export,'manifest',lambda *a,**k:rows)
    def fake_extract(path,preprocessing,split,augment=False):
        assert not augment
        return torch.arange((int(path)+3)*40,dtype=torch.float32).reshape(-1,40)/100
    monkeypatch.setattr(feature_export,'extract',fake_extract)
    out=tmp_path/'bundle'
    feature_export.export(run,'unused','metadata/actor_splits.csv',out,batch_size=3)
    return out,rows,model,fake_extract


def test_export_read_alignment_masks_and_standardization(bundle):
    out,rows,model,extract=bundle
    report=validate_bundle(out)
    assert report['samples']==4 and report['feature_dim']==128
    assert report['test_metrics_computed'] is False
    pooled=load_pooled(out,'train')
    assert pooled['pooled'].shape==(2,128)
    ids=pooled['sample_ids'][::-1].tolist()
    aligned=align_pooled(out,'train',ids,[2,2],['train','train'])
    np.testing.assert_array_equal(aligned,pooled['pooled'][::-1])
    with pytest.raises(ValueError): align_pooled(out,'train',ids,[1,2],['train','train'])
    with pytest.raises(ValueError): align_pooled(out,'train',ids,[2,2],['test','train'])
    with pytest.raises(ValueError): align_pooled(out,'train',['missing'],[2],['train'])
    with pytest.raises(ValueError): align_pooled(out,'train',[ids[0],ids[0]],[2,2],['train','train'])
    records=[load_sequence(out,r['sample_id']) for r in rows]
    padded=pad_sequences(records)
    assert padded['padding_mask'].dtype==np.bool_
    for i,record in enumerate(records):
        n=int(record['length'])
        assert not padded['padding_mask'][i,:n].any()
        assert padded['padding_mask'][i,n:].all()
        np.testing.assert_array_equal(padded['sequence'][i,:n],record['sequence'])
        row=rows[i]; features=extract(row['path'],{},row['split'])
        with torch.inference_mode(): expected=model(features[None],torch.tensor([len(features)]))
        np.testing.assert_allclose(expected['pooled'][0].numpy(),record['pooled'],rtol=2e-5,atol=2e-6)
    with pytest.raises(ValueError): pad_sequences([])
    metadata=json.loads((out/'metadata.json').read_text())
    assert metadata['augmentation'] is False and metadata['test_features_exported'] is True
    assert (out/'handoff_reader.py').exists()


def test_selection_guards(run):
    run,_=run
    feature_export.selected_checkpoint(run,'metadata/actor_splits.csv')
    selection=json.loads((run/'comparison.json').read_text())
    selection['selected_encoder']='cnn'
    write(run/'comparison.json',selection)
    with pytest.raises(ValueError,match='maximize'): feature_export.selected_checkpoint(run,'metadata/actor_splits.csv')
    selection['selected_encoder']='transformer'; selection['diagnostic']=True
    write(run/'comparison.json',selection)
    with pytest.raises(ValueError,match='formal'): feature_export.selected_checkpoint(run,'metadata/actor_splits.csv')


def test_corrupt_feature_detected(bundle):
    out,rows,model,extract=bundle
    target=out/'per_sample/train'/ (rows[0]['sample_id']+'.npz')
    original=target.read_bytes(); target.write_bytes(original+b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'): validate_bundle(out)
    target.write_bytes(original)
    validate_bundle(out)
    with pytest.raises(ValueError): load_sequence(out,'missing')


def test_existing_output_is_not_overwritten(run,tmp_path,monkeypatch):
    run,_=run
    monkeypatch.setattr(feature_export,'manifest',lambda *a,**k:[])
    out=tmp_path/'existing'; out.mkdir()
    sentinel=out/'keep.txt'; sentinel.write_text('keep')
    with pytest.raises(FileExistsError): feature_export.export(run,'unused','metadata/actor_splits.csv',out)
    assert sentinel.read_text()=='keep'


def test_delivery_is_portable_and_marks_m1_pending(bundle,tmp_path):
    import zipfile
    from ravdess_m2.delivery import build_delivery
    out,rows,model,extract=bundle
    delivery=tmp_path/'delivery'
    build_delivery(out,delivery,make_zip=True)
    assert (delivery/'best_audio_encoder.pt').is_file()
    assert (delivery/'audio_inference.py').is_file()
    assert (delivery/'ravdess_m2/inference.py').is_file()
    config=json.loads((delivery/'audio_config.json').read_text())
    assert config['encoder']=='transformer' and config['architecture']['hidden_dim']==128
    contract=json.loads((delivery/'stage5_contract.json').read_text())
    assert contract['outputs']['F_seq']['ready'] and contract['outputs']['F_learned']['ready']
    assert not contract['outputs']['F_audio']['ready'] and not contract['outputs']['P_audio']['ready']
    manifest=json.loads((delivery/'delivery_manifest.json').read_text())
    assert manifest['m1_final_audio_outputs_included'] is False
    assert contract['outputs']['F_audio_learned_feature_only']['ready']
    assert contract['outputs']['P_audio_learned_feature_only']['ready']
    assert contract['outputs']['F_sequence']['ready']
    assert manifest['m5_learned_only_interface']['samples']==len(rows)
    for row in rows:
        sample=delivery/'m5_learned_feature_only/per_sample'/row['split']/(row['sample_id']+'.npz')
        with np.load(sample,allow_pickle=False) as z:
            assert str(z['sample_id'])==row['sample_id'] and int(z['label'])==row['label']
            assert str(z['split'])==row['split']
            assert z['F_audio_learned_feature_only'].shape==(128,)
            assert z['P_audio_learned_feature_only'].shape==(8,)
            with torch.inference_mode():
                expected=model.head(torch.from_numpy(z['F_audio_learned_feature_only'])).softmax(-1).numpy()
            np.testing.assert_allclose(z['P_audio_learned_feature_only'],expected,rtol=1e-5,atol=1e-7)
            np.testing.assert_allclose(z['F_sequence'].mean(0),z['F_audio_learned_feature_only'],rtol=2e-5,atol=2e-6)
            assert np.isclose(z['P_audio_learned_feature_only'].sum(),1)
            assert not z['padding_mask'].any()
        aggregate=np.load(delivery/'m5_learned_feature_only/pooled'/(row['split']+'.npz'),allow_pickle=False)
        assert 'F_audio_learned_feature_only' in aggregate and 'P_audio_learned_feature_only' in aggregate

    for filename,digest in manifest['files_sha256'].items():
        assert feature_export.sha256(delivery/filename)==digest
    with zipfile.ZipFile(str(delivery)+'.zip') as z: assert z.testzip() is None
    with pytest.raises(FileExistsError): build_delivery(out,delivery,make_zip=True)
