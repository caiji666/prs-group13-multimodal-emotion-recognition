import numpy as np
import pytest
import torch
from ravdess_m2.augmentation_cache import EpochAugmentationCache
from ravdess_m2.train import TrainingDataset,validate_config

def test_train_only_cache_preserves_features(tmp_path,monkeypatch):
    import ravdess_m2.augmentation_cache as module
    rows=[]
    for i in range(2):
        p=tmp_path/f'{i}.wav'; p.write_bytes(b'fixture')
        rows.append(dict(sample_id=str(i),path=str(p),split='train',label=i))
    calls=[]
    def extract(path,config,split,augment,rng):
        calls.append(path); assert split=='train' and augment
        return torch.from_numpy(rng.normal(size=(4,40)).astype('float32'))
    monkeypatch.setattr(module,'extract',extract)
    cache=EpochAugmentationCache(rows,{'n_mfcc':40},42,tmp_path/'cache')
    a=cache.get(1,0)
    expected=np.random.default_rng(np.random.SeedSequence([42,1,0])).normal(size=(4,40)).astype('float32')
    np.testing.assert_array_equal(a.numpy(),expected)
    assert len(calls)==2
    second=EpochAugmentationCache(rows,{'n_mfcc':40},42,tmp_path/'cache')
    torch.testing.assert_close(a,second.get(1,0)); assert len(calls)==2
    with pytest.raises(ValueError): EpochAugmentationCache([dict(rows[0],split='val')],{'n_mfcc':40},42,tmp_path/'bad')
    clean={r['sample_id']:torch.zeros(4,40) for r in rows}
    dataset=TrainingDataset(rows,{},clean,'train',True,42)
    dataset.augment_probability=0
    assert torch.equal(dataset[0]['features'],clean['0'])


def test_five_experiment_summary_selects_validation_best(tmp_path):
    import json
    from ravdess_m2.tuning import summarize
    for i in range(1,6):
        run=tmp_path/f'round_{i:02d}'; run.mkdir()
        (run/'status.json').write_text('{"status":"completed"}')
        (run/'protocol.json').write_text('{"training":{"seed":42}}')
        rows=[]
        for kind in ['cnn','bilstm','transformer']:
            (run/kind).mkdir()
            rows.append(dict(encoder=kind,best_epoch=1,macro_f1=i/10,accuracy=i/10,macro_precision=i/10,macro_recall=i/10,checkpoint=f'{kind}/best.pt'))
            (run/kind/'summary.json').write_text('{"epochs_completed":2,"elapsed_seconds":1.0}')
        (run/'comparison.json').write_text(json.dumps(dict(selected_encoder='cnn',comparison=rows)))
    report=summarize(tmp_path)
    assert report['experiments_completed']==5 and report['models_completed']==15
    assert report['best']['experiment']=='round_05'
    assert (tmp_path/'suite_results.csv').exists()
