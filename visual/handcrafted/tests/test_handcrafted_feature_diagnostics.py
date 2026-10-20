"""Tests for the train-only handcrafted feature diagnostic (synthetic arrays in tmp_path)."""
import csv
import hashlib
import json

import numpy as np
import pytest
import sklearn.neural_network
import sklearn.svm

from m3 import diagnose_handcrafted_features as dg
from m3.common import EMOTIONS, ROOT

NAMES = [f"{b}_r{r}c{c}_{s}" for b in dg.SUBBANDS for r in range(4) for c in range(4) for s in dg.STATS]
COUNTS = {"train": 24, "val": 8, "test": 8}
COUNTS_ARG = "train=24,val=8,test=8"
ACTORS = {"train": [1, 2, 3, 4], "val": [5, 6], "test": [7, 8]}


def make_data(seed=0):
    """Synthetic export with exact per-frame energy = var + mean^2."""
    rng = np.random.default_rng(seed)
    ids, split = [], []
    for s, actors in ACTORS.items():
        per = COUNTS[s] // len(actors)
        for a in actors:
            for k in range(per):
                ids.append(f"01-{k % 8 + 1:02d}-01-{k // 8 + 1:02d}-01-{a:02d}")
                split.append(s)
    n = len(ids)
    S = rng.normal(size=(n, 16, 320)).astype(np.float64)
    idx = {name: i for i, name in enumerate(NAMES)}
    for b in dg.SUBBANDS:
        for r in range(4):
            for c in range(4):
                m = rng.normal(size=(n, 16))
                v = rng.uniform(0.1, 1.0, size=(n, 16))
                S[:, :, idx[f"{b}_r{r}c{c}_mean"]] = m
                S[:, :, idx[f"{b}_r{r}c{c}_var"]] = v
                S[:, :, idx[f"{b}_r{r}c{c}_energy"]] = v + m ** 2
    S = S.astype(np.float32)
    return {"sample_id": np.array(ids), "split": np.array(split),
            "label": np.arange(n) % 8, "class_names": np.array(EMOTIONS), "F_sequence": S,
            "F_visual_handcrafted": S.mean(axis=1), "feature_names": np.array(NAMES)}


def save(path, d):
    np.savez(path, **d)
    return path


def run(inp, out, *extra):
    return dg.main(["--input", str(inp), "--out-dir", str(out), "--expected-counts", COUNTS_ARG, *extra])


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("diag")
    inp = save(root / "in.npz", make_data())
    before = hashlib.sha256(inp.read_bytes()).hexdigest()
    out = root / "out"
    summary = run(inp, out)
    return {"root": root, "inp": inp, "out": out, "summary": summary, "hash": before}


# 1. real naming format parses into 4 x 16 x 5
def test_parse_names():
    schema = dg.parse_feature_names(NAMES)
    assert len(schema) == 320
    assert schema[6] == {"index": 6, "name": "LL_r0c1_var", "subband": "LL", "row": 0, "col": 1,
                         "stat": "var"}  # 0-4: LL_r0c0_*, 5: LL_r0c1_mean, 6: LL_r0c1_var
    assert {f["subband"] for f in schema} == set(dg.SUBBANDS)
    assert {f["stat"] for f in schema} == set(dg.STATS)


# 2. / 3. / 4. bad names, duplicates, incomplete structure
@pytest.mark.parametrize("bad", ["LL_r4c0_mean", "XX_r0c0_mean", "LL_r0c0_max", "LL-r0c0-mean"])
def test_invalid_name(bad):
    names = NAMES.copy()
    names[5] = bad
    with pytest.raises(dg.DiagnosticError, match="does not match"):
        dg.parse_feature_names(names)


def test_duplicate_and_incomplete_names():
    names = NAMES.copy()
    names[5] = names[4]
    with pytest.raises(dg.DiagnosticError, match="duplicate"):
        dg.parse_feature_names(names)
    with pytest.raises(dg.DiagnosticError, match="= 320"):
        dg.parse_feature_names(NAMES[:-5])


# 5. / 6. pooled mismatch, NaN / Inf, other structural checks
def test_input_checks(tmp_path):
    d = make_data()
    d["F_visual_handcrafted"] = d["F_visual_handcrafted"].copy()
    d["F_visual_handcrafted"][0, 0] += 0.5
    with pytest.raises(dg.DiagnosticError, match="differs from F_sequence"):
        dg.load_input(save(tmp_path / "a.npz", d), COUNTS)
    for bad in (np.nan, np.inf):
        d = make_data()
        d["F_sequence"] = d["F_sequence"].copy()
        d["F_sequence"][1, 2, 3] = bad
        with pytest.raises(dg.DiagnosticError, match="NaN"):
            dg.load_input(save(tmp_path / "b.npz", d), COUNTS)
    d = make_data()
    d["sample_id"] = d["sample_id"].astype("U32")
    d["sample_id"][0] = d["sample_id"][0] + "__aug01"
    with pytest.raises(dg.DiagnosticError, match="augmented"):
        dg.load_input(save(tmp_path / "c.npz", d), COUNTS)
    with pytest.raises(dg.DiagnosticError, match="split counts"):
        dg.load_input(save(tmp_path / "d.npz", make_data()), {"train": 960, "val": 240, "test": 240})
    d = make_data()
    del d["feature_names"]
    with pytest.raises(dg.DiagnosticError, match="missing fields"):
        dg.load_input(save(tmp_path / "e.npz", d), COUNTS)


# 7. only train values are used
def test_val_test_values_do_not_matter(base, tmp_path):
    d = make_data()
    nontrain = d["split"] != "train"
    d["F_sequence"] = d["F_sequence"].copy()
    d["F_sequence"][nontrain] = np.random.default_rng(9).normal(size=d["F_sequence"][nontrain].shape) * 100
    d["F_visual_handcrafted"] = d["F_sequence"].mean(axis=1)
    run(save(tmp_path / "x.npz", d), tmp_path / "o")
    for f in ("feature_statistics.csv", "high_correlation_pairs.csv", "variance_energy_analysis.csv",
              "statistics_type_summary.csv", "subband_summary.csv"):
        assert (tmp_path / "o" / f).read_text() == (base["out"] / f).read_text(), f


# 8. / 9. constant and near-zero features
def test_constant_and_near_zero():
    X = np.random.default_rng(0).normal(size=(30, 320))
    X[:, 3] = 2.5
    X[:, 10] = 1.0 + np.linspace(0, 1e-10, 30)
    schema = dg.parse_feature_names(NAMES)
    rows = dg.feature_statistics(X, schema)
    assert rows[3]["constant"] and rows[3]["near_zero_std_le_1e-12"]
    assert not rows[10]["constant"] and rows[10]["near_zero_std_le_1e-08"]
    assert rows[10]["near_zero_scale_aware"] and not rows[0]["near_zero_scale_aware"]
    assert sum(r["constant"] for r in rows) == 1


# 10. / 11. / 12. correlation: upper triangle, constants excluded, sign handled
def test_correlation_pairs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 320))
    X[:, 1] = X[:, 0] * 2 + 1e-6 * rng.normal(size=200)    # r ~ +1
    X[:, 2] = -X[:, 0] + 1e-6 * rng.normal(size=200)       # r ~ -1
    X[:, 5] = 7.0                                           # constant
    schema = dg.parse_feature_names(NAMES)
    constant = [bool(X[:, i].max() == X[:, i].min()) for i in range(320)]
    res = dg.correlation_analysis(X, schema, constant)
    keys = [(p["feature_a_index"], p["feature_b_index"]) for p in res["pairs"]]
    assert all(a < b for a, b in keys) and len(set(keys)) == len(keys)
    assert {(0, 1), (0, 2), (1, 2)} <= set(keys)
    assert res["counts"][">=0.9999"] == 3  # negative r counted by |r|
    assert any(p["correlation"] < 0 for p in res["pairs"])
    assert res["n_pairs_considered"] == 319 * 318 // 2
    assert not any(5 in k for k in keys) and res["excluded_constant_features"] == [NAMES[5]]
    assert np.isfinite([p["correlation"] for p in res["pairs"]]).all()


# 13. / 14. / 15. / 16. variance-energy groups and identities
def test_variance_energy_analysis():
    d = make_data()
    tr = d["split"] == "train"
    S = d["F_sequence"][tr].astype(np.float64)
    X = d["F_visual_handcrafted"][tr].astype(np.float64)
    schema = dg.parse_feature_names(NAMES)
    rows = dg.variance_energy_analysis(X, S, schema)
    assert len(rows) == 64
    for r in rows:
        assert r["var_name"].replace("_var", "") == r["energy_name"].replace("_energy", "")
        assert r["frame_identity_allclose"] and r["pooled_identity_allclose"]
        assert r["frame_observations"] == int(tr.sum()) * 16
        # the naive pooled formula is NOT the identity (mean_t(m^2) != mean_t(m)^2)
        assert r["naive_pooled_mean_sq_abs_residual_max"] > 1e-3
    # break one energy column -> residual detected for that group only
    S2 = S.copy()
    ie = NAMES.index("HL_r2c1_energy")
    S2[:, :, ie] += 0.01
    rows2 = dg.variance_energy_analysis(S2.mean(axis=1), S2, schema)
    broken = [r for r in rows2 if not r["frame_identity_allclose"]]
    assert [(r["subband"], r["grid_row"], r["grid_col"]) for r in broken] == [("HL", 2, 1)]
    assert broken[0]["frame_abs_residual_max"] == pytest.approx(0.01, rel=1e-3)


# 17. output files and fields
def test_outputs_complete(base):
    out = base["out"]
    for f in ("README.md", "diagnostic_summary.json", "feature_statistics.csv", "high_correlation_pairs.csv",
              "variance_energy_analysis.csv", "statistics_type_summary.csv", "subband_summary.csv"):
        assert (out / f).exists(), f
    assert not any(p.suffix in (".joblib", ".npz") for p in out.iterdir())
    s = json.loads((out / "diagnostic_summary.json").read_text())
    for k in ("input_path", "input_sha256", "analysis_split", "train_sample_count", "train_actor_count",
              "feature_count", "sequence_shape", "pooled_shape", "class_names", "feature_schema",
              "nan_count", "inf_count", "pooled_vs_sequence_mean_max_abs_diff", "constant_feature_count",
              "near_zero_counts", "correlation", "variance_energy", "counts_by_subband",
              "counts_by_statistic", "thresholds", "conclusion"):
        assert k in s, k
    assert s["analysis_split"] == "train" and s["train_sample_count"] == 24 and s["train_actor_count"] == 4
    for k in ("used_validation_values", "used_test_values", "features_removed", "models_trained",
              "formal_scores_produced"):
        assert s[k] is False
    assert s["input_sha256"] == base["hash"]
    stats = read_csv(out / "feature_statistics.csv")
    assert len(stats) == 320
    for k in ("feature_index", "feature_name", "subband", "grid_row", "grid_col", "statistic", "mean",
              "std", "variance", "min", "max", "max_abs", "range", "finite_count", "unique_count",
              "constant", "near_zero_std_le_1e-12", "near_zero_std_le_1e-08", "near_zero_std_le_1e-06",
              "near_zero_scale_aware"):
        assert k in stats[0]
    with open(out / "high_correlation_pairs.csv") as fh:
        assert next(csv.reader(fh)) == dg.PAIR_COLUMNS
    assert len(read_csv(out / "variance_energy_analysis.csv")) == 64
    assert {r["statistic_type"] for r in read_csv(out / "statistics_type_summary.csv")} == set(dg.STATS)
    assert {r["subband"] for r in read_csv(out / "subband_summary.csv")} == set(dg.SUBBANDS)
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "does not modify the feature set, train a classifier, select a model, or use validation/test" in readme


# 18. / 19. / 20. output folder protection
def test_out_dir_rules(base, tmp_path):
    with pytest.raises(SystemExit, match="already exists"):
        run(base["inp"], base["out"])
    run(base["inp"], base["out"], "--overwrite")  # own folder: allowed
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("x")
    with pytest.raises(SystemExit, match="not created by"):
        run(base["inp"], other, "--overwrite")
    assert (other / "keep.txt").read_text() == "x"
    for d in dg.PROTECTED_DIRS:
        with pytest.raises(SystemExit, match="Refusing"):
            run(base["inp"], ROOT / d / "diag")


def test_failed_input_writes_nothing(tmp_path):
    d = make_data()
    d["feature_names"] = np.array(NAMES[:-1] + [NAMES[0]])
    with pytest.raises(SystemExit, match="duplicate"):
        run(save(tmp_path / "bad.npz", d), tmp_path / "o")
    assert not (tmp_path / "o").exists()


# 21. no model training
def test_no_training(base, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no training allowed")

    monkeypatch.setattr(sklearn.svm.SVC, "fit", boom)
    monkeypatch.setattr(sklearn.neural_network.MLPClassifier, "fit", boom)
    run(base["inp"], tmp_path / "o")


# 22. reproducible
def test_reproducible(base, tmp_path):
    run(base["inp"], tmp_path / "again")
    for f in ("feature_statistics.csv", "high_correlation_pairs.csv", "variance_energy_analysis.csv",
              "statistics_type_summary.csv", "subband_summary.csv", "README.md"):
        assert (tmp_path / "again" / f).read_text() == (base["out"] / f).read_text(), f
    a = json.loads((base["out"] / "diagnostic_summary.json").read_text())
    b = json.loads((tmp_path / "again" / "diagnostic_summary.json").read_text())
    assert a == b
    assert hashlib.sha256(base["inp"].read_bytes()).hexdigest() == base["hash"]


# 23. real file: read-only smoke run (into tmp_path)
REAL = ROOT / dg.DEFAULT_INPUT


@pytest.mark.skipif(not REAL.exists(), reason="real handcrafted export not present")
def test_real_read_only_smoke(tmp_path):
    before = hashlib.sha256(REAL.read_bytes()).hexdigest()
    s = dg.main(["--input", str(REAL), "--out-dir", str(tmp_path / "real")])
    assert hashlib.sha256(REAL.read_bytes()).hexdigest() == before == s["input_sha256"]
    assert s["train_sample_count"] == 960 and s["train_actor_count"] == 16 and s["feature_count"] == 320
    assert s["variance_energy"]["groups"] == 64
    assert s["variance_energy"]["frame_identity_allclose_groups"] == 64
