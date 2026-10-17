"""Shared repository contracts and the shipped tuned checkpoint."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from ravdess_m2.data import LABELS, parse_filename, read_splits
from ravdess_m2.feature_export import selected_checkpoint, verify_source
from ravdess_m2.inference import AudioInference

ROOT=Path(__file__).resolve().parents[1]

def test_team_split_and_names():
    team=ROOT.parents[1]/'multimodal_fusion/configs'
    assert read_splits(ROOT/'metadata/actor_splits.csv')==read_splits(team/'actor_splits.csv')
    actors=json.loads((team/'actors_m2.json').read_text())
    assert read_splits(team/'actor_splits.csv')=={a:s for s,values in actors.items() for a in values}
    assert LABELS[-1]=='surprise'
    assert parse_filename('03-01-08-01-02-01-12.wav')['sample_id']=='01-08-01-02-01-12'

def test_selected_checkpoint_and_inference_weights():
    _,saved,_,protocol=selected_checkpoint(ROOT/'models/selected_run',ROOT/'metadata/actor_splits.csv')
    assert saved['encoder']=='transformer' and saved['class_names']==LABELS
    assert saved['validation_metrics']['accuracy']==0.5458333333333333
    inference=AudioInference(ROOT/'models/best_audio_encoder.pt')
    assert all(torch.equal(v,inference.model.state_dict()[k]) for k,v in saved['model_state_dict'].items())
    # The migration exception cannot admit other data-source edits.
    invalid=json.loads(json.dumps(protocol));invalid['source_sha256']['data.py']='0'*64
    import pytest
    with pytest.raises(ValueError):verify_source(invalid,'data.py')


def test_portable_refinement_and_replay_dependencies():
    import yaml
    config=yaml.safe_load((ROOT/'configs/tuning_20261017/best_resolved.yaml').read_text())
    parent=ROOT/config['warm_start']
    assert parent.resolve().is_relative_to(ROOT.resolve())
    assert json.loads((parent/'status.json').read_text())['status']=='completed'
    for kind in ('cnn','bilstm','transformer'):
        saved=torch.load(parent/kind/'best.pt',map_location='cpu',weights_only=True)
        assert saved['encoder']==kind and saved['class_names']==LABELS
    plan=json.loads((ROOT/'configs/tuning_20261017/plan.json').read_text())
    baseline=ROOT/plan['baseline']
    assert baseline.resolve().is_relative_to(ROOT.resolve())
    assert json.loads((baseline/'status.json').read_text())['status']=='completed'
    for kind in ('cnn','bilstm','transformer'):
        assert (baseline/kind/'best.pt').is_file()
        assert (baseline/kind/'best_val_metrics.json').is_file()
