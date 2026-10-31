"""Core tests for the M4 field standardisation (synthetic NPZ in tmp_path)."""
import numpy as np
import pytest

from m3 import standardize_m4_visual_outputs as sm
from m3.common import EMOTIONS


def make_source(path, n_per=(8, 2, 2), dim=6, drop=()):
    rng = np.random.default_rng(0)
    ids, split, label = [], [], []
    for s, k in zip(("train", "val", "test"), n_per):
        for a in range(k):
            for e in range(8):
                ids.append(f"01-{e + 1:02d}-01-01-01-{a + 1:02d}_{s}")
                split.append(s)
                label.append(e)
    n = len(ids)
    S = rng.normal(size=(n, 16, dim)).astype(np.float32)
    P = rng.random((n, 8)).astype(np.float32)
    d = {"sample_id": np.array(ids), "label": np.array(label, np.int32), "split": np.array(split),
         "sequence_features": S, "pooled_features": S.mean(axis=1), "probabilities": P / P.sum(1, keepdims=True),
         "class_names": np.array([c.capitalize() for c in EMOTIONS]),
         "source_model": np.array("MobileNetV3-Small (test)"), "source_checkpoint": np.array("x.keras")}
    for k in drop:
        d.pop(k)
    np.savez(path, **d)
    return d


def run(src, out, *extra):
    return sm.main(["--input", str(src), "--output", str(out), "--m3", "none.npz", *extra])


@pytest.fixture
def counts(monkeypatch):
    monkeypatch.setattr(sm, "EXPECTED_COUNTS", {"train": 64, "val": 16, "test": 16})


def test_mapping_and_lowercase(tmp_path, counts):
    src = make_source(tmp_path / "in.npz")
    run(tmp_path / "in.npz", tmp_path / "out.npz")
    z = np.load(tmp_path / "out.npz", allow_pickle=False)
    assert set(z.files) == set(sm.MAPPING.values())
    assert list(z["class_names"]) == EMOTIONS
    assert str(z["source_model"]) == "MobileNetV3-Small (test)" and str(z["source_checkpoint"]) == "x.keras"


def test_values_unchanged(tmp_path, counts):
    src = make_source(tmp_path / "in.npz")
    before = sm.sha256(tmp_path / "in.npz")
    run(tmp_path / "in.npz", tmp_path / "out.npz")
    z = np.load(tmp_path / "out.npz", allow_pickle=False)
    for old, new in sm.MAPPING.items():
        if old != "class_names":
            assert z[new].dtype == src[old].dtype and z[new].shape == src[old].shape
            np.testing.assert_array_equal(z[new], src[old])
    assert sm.sha256(tmp_path / "in.npz") == before


def test_missing_field_reported(tmp_path, counts):
    make_source(tmp_path / "in.npz", drop=("pooled_features",))
    with pytest.raises(SystemExit, match="missing fields.*pooled_features"):
        run(tmp_path / "in.npz", tmp_path / "out.npz")
    assert not (tmp_path / "out.npz").exists()


def test_existing_different_output_not_overwritten(tmp_path, counts):
    make_source(tmp_path / "in.npz")
    out = tmp_path / "out.npz"
    np.savez(out, other=np.zeros(3))
    with pytest.raises(SystemExit, match="different content"):
        run(tmp_path / "in.npz", out)
    assert list(np.load(out).files) == ["other"]
    assert run(tmp_path / "in.npz", out, "--overwrite")["status"] == "written"
    assert run(tmp_path / "in.npz", out)["status"].startswith("reused")
