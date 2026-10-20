"""Checks on the real exported file (skipped if the pipeline has not been run).

Set M3_BASELINE_DIR to a folder holding the pre-change dwt_features.npz,
m3_visual_handcrafted.npz and metrics.json to also check that features,
probabilities and metrics are unchanged.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pytest

from m3.common import (EMOTIONS, ROOT, actor_from_sample_id, label_from_sample_id,
                       load_actor_splits)

OUT = ROOT / "outputs" / "m3_visual_handcrafted.npz"
pytestmark = pytest.mark.skipif(not OUT.exists(), reason="pipeline outputs not present")


@pytest.fixture(scope="module")
def out():
    return dict(np.load(OUT, allow_pickle=False))


# 7. all 1440 samples have timing fields
def test_timing_fields_present(out):
    assert len(out["sample_id"]) == 1440
    assert out["frame_idx"].shape == (1440, 16)
    assert out["frame_time_sec"].shape == (1440, 16)
    assert out["video_fps"].shape == (1440,)


# 8. timestamps finite and non-negative; fps finite and positive
def test_timing_values_valid(out):
    t, fps = out["frame_time_sec"], out["video_fps"]
    assert np.isfinite(t).all() and (t >= 0).all()
    assert np.isfinite(fps).all() and (fps > 0).all()
    np.testing.assert_allclose(t, out["frame_idx"] / fps[:, None], rtol=1e-6, atol=1e-6)


# 9. non-decreasing per sample
def test_timing_non_decreasing(out):
    assert (np.diff(out["frame_idx"], axis=1) >= 0).all()
    assert (np.diff(out["frame_time_sec"], axis=1) >= 0).all()


# 10. ids, split, label and class order unchanged
def test_ids_split_label_class_order(out):
    splits = load_actor_splits()
    ids = out["sample_id"]
    assert len(set(ids)) == 1440
    assert [splits[actor_from_sample_id(s)] for s in ids] == list(out["split"])
    assert [label_from_sample_id(s) for s in ids] == list(out["label"])
    assert list(out["class_names"]) == EMOTIONS
    assert {s: int((out["split"] == s).sum()) for s in ("train", "val", "test")} == \
        {"train": 960, "val": 240, "test": 240}


def test_joblib_matches_exported_probabilities(out):
    model = joblib.load(ROOT / "outputs" / "dwt_svm.joblib")
    P = model.predict_proba(out["F_visual_handcrafted"])
    np.testing.assert_allclose(P, out["P_visual_handcrafted"], atol=1e-6)


# 6. features / probabilities / metrics unchanged versus the pre-change snapshot
BASE = os.environ.get("M3_BASELINE_DIR")


@pytest.mark.skipif(not BASE, reason="M3_BASELINE_DIR not set")
def test_unchanged_versus_baseline(out):
    base_out = np.load(Path(BASE) / "m3_visual_handcrafted.npz")
    base_feat = np.load(Path(BASE) / "dwt_features.npz")
    feat = np.load(ROOT / "data" / "dwt_features.npz")
    for k in ("sample_id", "split", "label", "class_names", "feature_names", "sequence_length",
              "padding_mask", "F_sequence", "F_visual_handcrafted"):
        np.testing.assert_array_equal(out[k], base_out[k], err_msg=k)
    for k in ("sample_id", "split", "label", "F_sequence", "F_pooled", "feature_names"):
        np.testing.assert_array_equal(feat[k], base_feat[k], err_msg=k)
    np.testing.assert_allclose(out["P_visual_handcrafted"], base_out["P_visual_handcrafted"], atol=1e-6)

    m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
    b = json.loads((Path(BASE) / "metrics.json").read_text())
    assert m["best_params"] == b["best_params"]
    for s in ("val", "test"):
        assert m[s]["macro_f1"] == b[s]["macro_f1"] and m[s]["accuracy"] == b[s]["accuracy"]
        assert m[s]["confusion_matrix"] == b[s]["confusion_matrix"]
