"""Read-only, train-only health and redundancy diagnostics of the 320 handcrafted DWT features.

Input: the M3 export (default outputs_augmentation/m3_visual_handcrafted_augmented.npz).
All numeric diagnostics use ONLY the 960 original train rows (split == "train"); the other
splits are read only to check shapes, ids and counts. Nothing is removed, no classifier is
trained, no model is selected and no score is produced. AUG-1 features and M4 features are
not read.

Outputs (--out-dir, default outputs_handcrafted_diagnostics/):
    README.md, diagnostic_summary.json, feature_statistics.csv, high_correlation_pairs.csv,
    variance_energy_analysis.csv, statistics_type_summary.csv, subband_summary.csv

Usage:
    uv run python -m m3.diagnose_handcrafted_features --input outputs_augmentation/m3_visual_handcrafted_augmented.npz --out-dir outputs_handcrafted_diagnostics
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from m3.common import ROOT

TOOL = "m3.diagnose_handcrafted_features"
DEFAULT_INPUT = "outputs_augmentation/m3_visual_handcrafted_augmented.npz"
DEFAULT_OUT = "outputs_handcrafted_diagnostics"
PROTECTED_DIRS = ("data", "outputs", "outputs_v1_global20", "outputs_ablation",
                  "outputs_augmentation", "outputs_visual_comparison", "external")
EXPECTED_COUNTS = {"train": 960, "val": 240, "test": 240}
N_FRAMES = 16
SUBBANDS = ("LL", "LH", "HL", "HH")
STATS = ("mean", "var", "energy", "entropy", "edge")
GRID = 4
NAME_RE = re.compile(r"^(LL|LH|HL|HH)_r([0-3])c([0-3])_(mean|var|energy|entropy|edge)$")
AUG_MARKERS = ("__aug", "aug01", "augmented")

# Fixed before looking at any result; reported, never used to remove features.
NEAR_ZERO_ABS = (1e-12, 1e-8, 1e-6)          # std <= t
NEAR_ZERO_SCALE = 1e-8                        # std <= t * max(1, |mean|, max_abs)
CORR_THRESHOLDS = (0.95, 0.99, 0.999, 0.9999)  # |r| >= t
CORR_LIST_THRESHOLD = 0.95
# pooled vs float64 sequence mean: |diff| <= atol + rtol * mean_t(|frame value|) (float32 pooling)
POOLED_RTOL, POOLED_ATOL = 1e-5, 1e-6
IDENTITY_RTOL, IDENTITY_ATOL = 1e-5, 1e-6     # float32 features
# Neutral-conclusion rule (fixed in advance): "substantial near-duplication" means any constant
# feature, or at least 16 features (5% of 320) involved in pairs with |r| >= 0.9999.
NEAR_DUPLICATE_R = 0.9999
NEAR_DUPLICATE_FEATURE_COUNT = 16


class DiagnosticError(ValueError):
    pass


# ------------------------------------------------------------------ input

def parse_feature_names(names):
    """Parse {subband}_r{row}c{col}_{stat}; require exactly 4 x 4x4 x 5 = 320 unique names."""
    names = [str(n) for n in names]
    if len(set(names)) != len(names):
        seen, dup = set(), []
        for n in names:
            if n in seen:
                dup.append(n)
            seen.add(n)
        raise DiagnosticError(f"duplicate feature names: {dup[:5]}")
    schema = []
    for i, n in enumerate(names):
        m = NAME_RE.match(n)
        if not m:
            raise DiagnosticError(f"feature {i} name {n!r} does not match "
                                  "{LL,LH,HL,HH}_r{0-3}c{0-3}_{mean,var,energy,entropy,edge}")
        schema.append({"index": i, "name": n, "subband": m.group(1), "row": int(m.group(2)),
                       "col": int(m.group(3)), "stat": m.group(4)})
    expected = {(b, r, c, s) for b in SUBBANDS for r in range(GRID) for c in range(GRID) for s in STATS}
    found = {(f["subband"], f["row"], f["col"], f["stat"]) for f in schema}
    if found != expected or len(schema) != len(expected):
        missing = sorted(expected - found)
        raise DiagnosticError(f"feature schema is not 4 subbands x 4x4 cells x 5 stats = 320 "
                              f"(got {len(schema)}; missing e.g. {missing[:5]})")
    return schema


def load_input(path, expected_counts=EXPECTED_COUNTS):
    path = Path(path)
    if not path.exists():
        raise DiagnosticError(f"input not found: {path}")
    z = np.load(path, allow_pickle=False)
    need = ("sample_id", "split", "label", "class_names", "F_sequence", "F_visual_handcrafted",
            "feature_names")
    missing = [k for k in need if k not in z.files]
    if missing:
        raise DiagnosticError(f"{path}: missing fields {missing}; actual fields {sorted(z.files)}")
    ids = z["sample_id"].astype(str)
    n = len(ids)
    if len(set(ids)) != n:
        raise DiagnosticError("duplicate sample_id")
    bad = [s for s in ids if any(m in s.lower() for m in AUG_MARKERS)]
    if bad:
        raise DiagnosticError(f"augmented sample_id not allowed, e.g. {bad[:3]}")
    split = z["split"].astype(str)
    counts = {str(s): int(c) for s, c in zip(*np.unique(split, return_counts=True))}
    if counts != expected_counts:
        raise DiagnosticError(f"split counts {counts}, expected {expected_counts}")
    S, F = z["F_sequence"], z["F_visual_handcrafted"]
    schema = parse_feature_names(z["feature_names"])
    d = len(schema)
    if S.shape != (n, N_FRAMES, d):
        raise DiagnosticError(f"F_sequence shape {S.shape}, expected ({n}, {N_FRAMES}, {d})")
    if F.shape != (n, d):
        raise DiagnosticError(f"F_visual_handcrafted shape {F.shape}, expected ({n}, {d})")
    for name, X in (("F_sequence", S), ("F_visual_handcrafted", F)):
        if not np.isfinite(X).all():
            raise DiagnosticError(f"{name}: {int(np.isnan(X).sum())} NaN and "
                                  f"{int(np.isinf(X).sum())} Inf values")
    S64 = S.astype(np.float64)
    seq_mean = S64.mean(axis=1)
    abs_diff = np.abs(seq_mean - F.astype(np.float64))
    diff = float(abs_diff.max())
    # float32 pooling error scales with the frame values, not with their (possibly ~0) mean
    if (abs_diff > POOLED_ATOL + POOLED_RTOL * np.abs(S64).mean(axis=1)).any():
        raise DiagnosticError(f"F_visual_handcrafted differs from F_sequence.mean(axis=1) "
                              f"(max |diff| {diff:.3g}; rtol {POOLED_RTOL}, atol {POOLED_ATOL})")
    return {"path": path, "sample_id": ids, "split": split, "label": z["label"],
            "class_names": [str(c) for c in z["class_names"]], "F_sequence": S,
            "F": F, "schema": schema, "pooled_vs_sequence_max_diff": diff}


def train_view(data):
    """The only rows used for numeric diagnostics."""
    tr = data["split"] == "train"
    return data["F"][tr].astype(np.float64), data["F_sequence"][tr].astype(np.float64), tr


# -------------------------------------------------------------- statistics

def feature_statistics(X, schema):
    rows = []
    for f in schema:
        x = X[:, f["index"]]
        mean, std = float(x.mean()), float(x.std())
        mn, mx = float(x.min()), float(x.max())
        max_abs = float(np.abs(x).max())
        row = {"feature_index": f["index"], "feature_name": f["name"], "subband": f["subband"],
               "grid_row": f["row"], "grid_col": f["col"], "statistic": f["stat"],
               "mean": mean, "std": std, "variance": float(x.var()), "min": mn, "max": mx,
               "max_abs": max_abs, "range": mx - mn, "finite_count": int(np.isfinite(x).sum()),
               "unique_count": int(np.unique(x).size), "constant": bool(mx == mn)}
        for t in NEAR_ZERO_ABS:
            row[f"near_zero_std_le_{t:g}"] = bool(std <= t)
        row["near_zero_scale_aware"] = bool(std <= NEAR_ZERO_SCALE * max(1.0, abs(mean), max_abs))
        rows.append(row)
    return rows


def _pair_info(fa, fb):
    same_cell = (fa["subband"], fa["row"], fa["col"]) == (fb["subband"], fb["row"], fb["col"])
    return {"same_subband": fa["subband"] == fb["subband"], "same_cell": same_cell,
            "same_statistic": fa["stat"] == fb["stat"],
            "var_energy_same_cell": same_cell and {fa["stat"], fb["stat"]} == {"var", "energy"}}


def correlation_analysis(X, schema, constant):
    """Pearson r over non-constant train columns, upper triangle only (i < j)."""
    keep = np.flatnonzero(~np.asarray(constant))
    R = np.corrcoef(X[:, keep], rowvar=False) if keep.size > 1 else np.ones((keep.size, keep.size))
    iu = np.triu_indices(keep.size, k=1)
    r = R[iu]
    absr = np.abs(r)
    counts = {f">={t:g}": int((absr >= t).sum()) for t in CORR_THRESHOLDS}
    order = np.argsort(-absr, kind="stable")
    pairs = []
    for k in order:
        if absr[k] < CORR_LIST_THRESHOLD:
            break
        a, b = keep[iu[0][k]], keep[iu[1][k]]
        fa, fb = schema[a], schema[b]
        pairs.append({"feature_a_index": int(a), "feature_a_name": fa["name"],
                      "feature_a_subband": fa["subband"], "feature_a_statistic": fa["stat"],
                      "feature_a_cell": f"r{fa['row']}c{fa['col']}",
                      "feature_b_index": int(b), "feature_b_name": fb["name"],
                      "feature_b_subband": fb["subband"], "feature_b_statistic": fb["stat"],
                      "feature_b_cell": f"r{fb['row']}c{fb['col']}",
                      "correlation": float(r[k]), "absolute_correlation": float(absr[k]),
                      **_pair_info(fa, fb)})
    involvement = np.zeros(len(schema), dtype=np.int64)
    near_dup = set()
    for p in pairs:
        involvement[p["feature_a_index"]] += 1
        involvement[p["feature_b_index"]] += 1
        if p["absolute_correlation"] >= NEAR_DUPLICATE_R:
            near_dup |= {p["feature_a_index"], p["feature_b_index"]}
    return {"pairs": pairs, "counts": counts, "n_pairs_considered": int(r.size),
            "max_abs_correlation": float(absr.max()) if r.size else None,
            "excluded_constant_features": [schema[i]["name"] for i in np.flatnonzero(constant)],
            "involvement": involvement, "near_duplicate_features": sorted(near_dup)}


def _abs_stats(res):
    a = np.abs(res).ravel()
    return {"max": float(a.max()), "mean": float(a.mean()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99))}


def variance_energy_analysis(X, S, schema):
    """64 groups (subband x cell): per-frame identity and pooled relations, train only.

    per frame:  energy = var + mean^2
    pooled:     pooled_energy = pooled_var + mean_over_frames(frame_mean^2)
                (NOT pooled_var + pooled_mean^2, since mean_t(m_t^2) != mean_t(m_t)^2)
    """
    idx = {(f["subband"], f["row"], f["col"], f["stat"]): f["index"] for f in schema}
    rows = []
    for b in SUBBANDS:
        for r in range(GRID):
            for c in range(GRID):
                im, iv, ie = (idx[(b, r, c, s)] for s in ("mean", "var", "energy"))
                m, v, e = S[:, :, im], S[:, :, iv], S[:, :, ie]
                frame_res = e - (v + m ** 2)
                fs = _abs_stats(frame_res)
                pooled_res = X[:, ie] - (X[:, iv] + (m ** 2).mean(axis=1))
                naive_res = X[:, ie] - (X[:, iv] + X[:, im] ** 2)
                pv, pe = X[:, iv], X[:, ie]
                corr = float(np.corrcoef(pv, pe)[0, 1]) if pv.std() > 0 and pe.std() > 0 else float("nan")
                mad = float(np.abs(pe - pv).mean())
                row = {"subband": b, "grid_row": r, "grid_col": c,
                       "mean_index": im, "var_index": iv, "energy_index": ie,
                       "var_name": schema[iv]["name"], "energy_name": schema[ie]["name"],
                       "frame_observations": int(frame_res.size),
                       **{f"frame_abs_residual_{k}": v_ for k, v_ in fs.items()},
                       "frame_identity_allclose": bool(np.allclose(e, v + m ** 2, rtol=IDENTITY_RTOL,
                                                                    atol=IDENTITY_ATOL)),
                       "pooled_identity_abs_residual_max": float(np.abs(pooled_res).max()),
                       "pooled_identity_allclose": bool(np.allclose(
                           X[:, ie], X[:, iv] + (m ** 2).mean(axis=1), rtol=IDENTITY_RTOL,
                           atol=IDENTITY_ATOL)),
                       "naive_pooled_mean_sq_abs_residual_max": float(np.abs(naive_res).max()),
                       "pooled_var_energy_correlation": corr,
                       "pooled_var_energy_abs_correlation": abs(corr),
                       "pooled_mean_abs_difference": mad,
                       "pooled_normalized_difference": mad / max(float(np.abs(pe).mean()), 1e-300),
                       "pooled_columns_identical": bool(np.array_equal(pv, pe))}
                for t in CORR_THRESHOLDS:
                    row[f"abs_corr_ge_{t:g}"] = bool(abs(corr) >= t)
                rows.append(row)
    return rows


def group_summary(stats_rows, involvement, key):
    out = []
    values = SUBBANDS if key == "subband" else STATS
    for val in values:
        rows = [r for r in stats_rows if r[key] == val]
        std = np.array([r["std"] for r in rows])
        out.append({key: val, "feature_count": len(rows),
                    "median_feature_mean": float(np.median([r["mean"] for r in rows])),
                    "median_feature_std": float(np.median(std)), "min_feature_std": float(std.min()),
                    "max_feature_std": float(std.max()),
                    "median_max_abs": float(np.median([r["max_abs"] for r in rows])),
                    "constant_count": sum(r["constant"] for r in rows),
                    "near_zero_scale_aware_count": sum(r["near_zero_scale_aware"] for r in rows),
                    "near_zero_std_le_1e-06_count": sum(r["near_zero_std_le_1e-06"] for r in rows),
                    "high_corr_pair_involvement": int(sum(involvement[r["feature_index"]] for r in rows))})
    if key == "statistic":
        for o in out:
            o["statistic_type"] = o.pop("statistic")
    return out


# ------------------------------------------------------------------ run

def run_diagnostics(data):
    X, S, tr = train_view(data)
    schema = data["schema"]
    stats = feature_statistics(X, schema)
    constant = [r["constant"] for r in stats]
    corr = correlation_analysis(X, schema, constant)
    ve = variance_energy_analysis(X, S, schema)
    for r in stats:
        r["high_corr_pair_involvement"] = int(corr["involvement"][r["feature_index"]])
    actors = sorted({int(s.split("-")[-1]) for s in data["sample_id"][tr]})  # last id field
    return {"stats": stats, "corr": corr, "ve": ve, "train_rows": int(tr.sum()),
            "train_actors": actors,
            "by_stat": group_summary(stats, corr["involvement"], "statistic"),
            "by_subband": group_summary(stats, corr["involvement"], "subband")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def conclusion(res):
    n_const = sum(r["constant"] for r in res["stats"])
    n_dup = len(res["corr"]["near_duplicate_features"])
    substantial = n_const > 0 or n_dup >= NEAR_DUPLICATE_FEATURE_COUNT
    if substantial:
        text = ("Substantial near-duplication was found by the pre-defined rule; consider "
                "discussing a pre-defined reduced handcrafted control before the formal Level-2 "
                "run. No feature is removed here, and test results must not decide any removal.")
    else:
        text = ("No substantial near-duplication by the pre-defined rule; there is not enough "
                "evidence to add a reduced handcrafted experiment.")
    return {"substantial_near_duplication": substantial, "constant_features": n_const,
            "features_in_near_duplicate_pairs": n_dup, "text": text}


def build_summary(data, res, input_hash):
    stats, corr, ve = res["stats"], res["corr"], res["ve"]
    S, F = data["F_sequence"], data["F"]
    return {
        "tool": TOOL, "input_path": str(data["path"]), "input_sha256": input_hash,
        "analysis_split": "train", "train_sample_count": res["train_rows"],
        "train_actor_count": len(res["train_actors"]), "train_actors": res["train_actors"],
        "feature_count": len(data["schema"]), "sequence_shape": list(S.shape),
        "pooled_shape": list(F.shape), "train_sequence_shape": [res["train_rows"], *S.shape[1:]],
        "train_pooled_shape": [res["train_rows"], F.shape[1]], "class_names": data["class_names"],
        "feature_schema": {"format": "{subband}_r{row}c{col}_{stat}", "subbands": SUBBANDS,
                           "grid": f"{GRID}x{GRID}", "statistics": STATS,
                           "count": f"{len(SUBBANDS)} x {GRID * GRID} x {len(STATS)} = {len(data['schema'])}"},
        "nan_count": int(np.isnan(S).sum() + np.isnan(F).sum()),
        "inf_count": int(np.isinf(S).sum() + np.isinf(F).sum()),
        "pooled_vs_sequence_mean_max_abs_diff": data["pooled_vs_sequence_max_diff"],
        "constant_feature_count": sum(r["constant"] for r in stats),
        "near_zero_counts": {**{f"std<={t:g}": sum(r[f"near_zero_std_le_{t:g}"] for r in stats)
                                for t in NEAR_ZERO_ABS},
                             "scale_aware": sum(r["near_zero_scale_aware"] for r in stats)},
        "correlation": {"method": "Pearson, train rows only, constant features excluded, upper "
                                  "triangle (i < j)",
                        "pairs_considered": corr["n_pairs_considered"],
                        "pair_counts_abs_r": corr["counts"],
                        "max_abs_correlation": corr["max_abs_correlation"],
                        "top_pair": corr["pairs"][0] if corr["pairs"] else None,
                        "var_energy_same_cell_pair_counts_abs_r": {
                            f">={t:g}": sum(p["var_energy_same_cell"] and p["absolute_correlation"] >= t
                                            for p in corr["pairs"]) for t in CORR_THRESHOLDS},
                        "excluded_constant_features": corr["excluded_constant_features"]},
        "variance_energy": {
            "groups": len(ve),
            "frame_identity": "energy = var + mean^2 per frame (train F_sequence)",
            "frame_identity_allclose_groups": sum(r["frame_identity_allclose"] for r in ve),
            "frame_max_abs_residual": max(r["frame_abs_residual_max"] for r in ve),
            "pooled_identity": "pooled_energy = pooled_var + mean_over_frames(frame_mean^2)",
            "pooled_identity_allclose_groups": sum(r["pooled_identity_allclose"] for r in ve),
            "pooled_identity_max_abs_residual": max(r["pooled_identity_abs_residual_max"] for r in ve),
            "naive_pooled_mean_sq_max_abs_residual": max(r["naive_pooled_mean_sq_abs_residual_max"] for r in ve),
            "pooled_abs_correlation_min": min(r["pooled_var_energy_abs_correlation"] for r in ve),
            "pooled_abs_correlation_median": float(np.median([r["pooled_var_energy_abs_correlation"] for r in ve])),
            "groups_abs_corr_ge": {f"{t:g}": sum(r[f"abs_corr_ge_{t:g}"] for r in ve) for t in CORR_THRESHOLDS},
            "groups_with_identical_pooled_columns": sum(r["pooled_columns_identical"] for r in ve),
            "by_subband": {b: {"median_abs_correlation": float(np.median(
                [r["pooled_var_energy_abs_correlation"] for r in ve if r["subband"] == b])),
                "groups_abs_corr_ge_0.9999": sum(r["abs_corr_ge_0.9999"] for r in ve if r["subband"] == b)}
                for b in SUBBANDS},
        },
        "counts_by_subband": {b: sum(f["subband"] == b for f in data["schema"]) for b in SUBBANDS},
        "counts_by_statistic": {s: sum(f["stat"] == s for f in data["schema"]) for s in STATS},
        "thresholds": {"constant": "max == min", "near_zero_abs_std": NEAR_ZERO_ABS,
                       "near_zero_scale_aware": f"std <= {NEAR_ZERO_SCALE:g} * max(1, |mean|, max_abs)",
                       "correlation_abs": CORR_THRESHOLDS, "listed_pairs_abs_r_ge": CORR_LIST_THRESHOLD,
                       "pooled_vs_sequence_rtol": POOLED_RTOL,
                       "pooled_vs_sequence_atol": POOLED_ATOL, "identity_rtol": IDENTITY_RTOL,
                       "identity_atol": IDENTITY_ATOL,
                       "near_duplicate_rule": f"constant features > 0 or >= {NEAR_DUPLICATE_FEATURE_COUNT} "
                                              f"features in pairs with |r| >= {NEAR_DUPLICATE_R}"},
        "conclusion": conclusion(res),
        "used_validation_values": False, "used_test_values": False, "used_aug1_features": False,
        "used_m4_features": False, "features_removed": False, "models_trained": False,
        "formal_scores_produced": False,
    }


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


PAIR_COLUMNS = ["feature_a_index", "feature_a_name", "feature_a_subband", "feature_a_statistic",
                "feature_a_cell", "feature_b_index", "feature_b_name", "feature_b_subband",
                "feature_b_statistic", "feature_b_cell", "correlation", "absolute_correlation",
                "same_subband", "same_cell", "same_statistic", "var_energy_same_cell"]


def write_readme(path, summary, res):
    s, ve, corr = summary, summary["variance_energy"], summary["correlation"]
    lines = [
        "# Handcrafted feature diagnostics (train-only, read-only)", "",
        "This diagnostic does not modify the feature set, train a classifier, select a model, "
        "or use validation/test feature values.", "",
        f"- Input: `{s['input_path']}`", f"- SHA-256: `{s['input_sha256']}`",
        f"- Rows analysed: train only, {s['train_sample_count']} clips from "
        f"{s['train_actor_count']} actors; pooled {s['train_pooled_shape']}, sequence "
        f"{s['train_sequence_shape']} (full file: {s['pooled_shape']} / {s['sequence_shape']}; "
        "val/test read only for shape checks).",
        f"- Schema: {s['feature_schema']['count']} (`{s['feature_schema']['format']}`).",
        f"- NaN: {s['nan_count']}, Inf: {s['inf_count']}; max |pooled - sequence mean| = "
        f"{s['pooled_vs_sequence_mean_max_abs_diff']:.3g}.",
        f"- Constant features: {s['constant_feature_count']}; near-zero std: "
        + ", ".join(f"{k}: {v}" for k, v in s["near_zero_counts"].items()) + ".", "",
        "## Correlation (Pearson, train, |r|, upper triangle)", "",
        f"{corr['pairs_considered']} pairs considered; constant features excluded: "
        f"{corr['excluded_constant_features'] or 'none'}.", "",
        "| Threshold | Pairs | of which same-cell var-energy |", "|---|---|---|",
        *[f"| \\|r\\| {k} | {v} | {corr['var_energy_same_cell_pair_counts_abs_r'][k]} |"
          for k, v in corr["pair_counts_abs_r"].items()], "",
        f"Highest |r| = {corr['max_abs_correlation']:.8f}. Top 20 pairs "
        "(full list in `high_correlation_pairs.csv`):", "",
        "| Feature A | Feature B | r | same cell | var-energy same cell |", "|---|---|---|---|---|",
        *[f"| {p['feature_a_name']} | {p['feature_b_name']} | {p['correlation']:.8f} | "
          f"{p['same_cell']} | {p['var_energy_same_cell']} |" for p in res["corr"]["pairs"][:20]],
        "", "## Variance vs energy (64 subband x cell groups)", "",
        f"- **Per frame (mathematical):** energy = var + mean^2 holds in "
        f"{ve['frame_identity_allclose_groups']}/64 groups (rtol {s['thresholds']['identity_rtol']}, "
        f"atol {s['thresholds']['identity_atol']}); max |residual| = {ve['frame_max_abs_residual']:.3g}.",
        f"- **After temporal pooling:** pooled_energy = pooled_var + mean_t(frame_mean^2) holds in "
        f"{ve['pooled_identity_allclose_groups']}/64 groups (max |residual| "
        f"{ve['pooled_identity_max_abs_residual']:.3g}). Using pooled_mean^2 instead would be wrong "
        f"(max |residual| {ve['naive_pooled_mean_sq_max_abs_residual']:.3g}).",
        f"- **Pooled statistical correlation:** median |r(var, energy)| = "
        f"{ve['pooled_abs_correlation_median']:.6f}, minimum {ve['pooled_abs_correlation_min']:.6f}; "
        f"groups with |r| >= threshold: {ve['groups_abs_corr_ge']}; groups whose pooled var and "
        f"energy columns are identical: {ve['groups_with_identical_pooled_columns']}.",
        "- Per subband median |r|: " + ", ".join(
            f"{b} {v['median_abs_correlation']:.6f}" for b, v in ve["by_subband"].items()) + ".",
        "- A high correlation is a statistical relation over clips; it does not mean the two pooled "
        "columns are identical.", "",
        "## Scale by statistic type and subband", "",
        "| Statistic | Count | Median mean | Median std | Min std | Max std | Constant | Near-zero (scale-aware) | High-corr involvement |",
        "|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['statistic_type']} | {r['feature_count']} | {r['median_feature_mean']:.4g} | "
          f"{r['median_feature_std']:.4g} | {r['min_feature_std']:.4g} | {r['max_feature_std']:.4g} | "
          f"{r['constant_count']} | {r['near_zero_scale_aware_count']} | {r['high_corr_pair_involvement']} |"
          for r in res["by_stat"]], "",
        "| Subband | Count | Median mean | Median std | Min std | Max std | Constant | Near-zero (scale-aware) | High-corr involvement |",
        "|---|---|---|---|---|---|---|---|---|",
        *[f"| {r['subband']} | {r['feature_count']} | {r['median_feature_mean']:.4g} | "
          f"{r['median_feature_std']:.4g} | {r['min_feature_std']:.4g} | {r['max_feature_std']:.4g} | "
          f"{r['constant_count']} | {r['near_zero_scale_aware_count']} | {r['high_corr_pair_involvement']} |"
          for r in res["by_subband"]], "",
        "## Conclusion (rule fixed in advance)", "",
        f"Rule: {s['thresholds']['near_duplicate_rule']}. Constant features: "
        f"{s['conclusion']['constant_features']}; features in |r| >= {NEAR_DUPLICATE_R} pairs: "
        f"{s['conclusion']['features_in_near_duplicate_pairs']}.", "",
        s["conclusion"]["text"], "",
        "No feature was removed. Validation and test feature values were not used, and test results "
        "must never decide which features to remove.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_out_dir(out_dir, input_path, overwrite):
    out = Path(out_dir).resolve()
    for d in PROTECTED_DIRS:
        p = (ROOT / d).resolve()
        if out == p or p in out.parents:
            raise SystemExit(f"Refusing to write diagnostics into {out} (protected: {p})")
    if out == ROOT.resolve() or out in Path(input_path).resolve().parents:
        raise SystemExit(f"Refusing to write diagnostics into {out}")
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} already exists; pass --overwrite or choose a new --out-dir")
        marker = out / "diagnostic_summary.json"
        try:
            ours = json.loads(marker.read_text()).get("tool") == TOOL
        except (OSError, ValueError):
            ours = False
        if not ours:
            raise SystemExit(f"{out} was not created by {TOOL}; --overwrite only replaces this "
                             "tool's own output folder")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an output folder previously written by this tool")
    ap.add_argument("--expected-counts", default="train=960,val=240,test=240")
    args = ap.parse_args(argv)
    counts = {k: int(v) for k, v in (p.split("=") for p in args.expected_counts.split(","))}
    out_dir = Path(args.out_dir)
    check_out_dir(out_dir, args.input, args.overwrite)
    try:
        data = load_input(args.input, counts)
    except DiagnosticError as e:
        raise SystemExit(f"ERROR: {e}") from None
    input_hash = sha256(args.input)
    res = run_diagnostics(data)
    summary = build_summary(data, res, input_hash)
    if sha256(args.input) != input_hash:
        raise SystemExit("ERROR: input file changed during the diagnostic")

    # All results computed; only now is the output folder written.
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "feature_statistics.csv", res["stats"])
    with open(out_dir / "high_correlation_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_COLUMNS)
        w.writeheader()
        w.writerows(res["corr"]["pairs"])
    write_csv(out_dir / "variance_energy_analysis.csv", res["ve"])
    write_csv(out_dir / "statistics_type_summary.csv", res["by_stat"])
    write_csv(out_dir / "subband_summary.csv", res["by_subband"])
    with open(out_dir / "diagnostic_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    write_readme(out_dir / "README.md", summary, res)
    print(f"Train-only diagnostic of {len(data['schema'])} features on {res['train_rows']} train rows "
          f"-> {out_dir}")
    print(f"  constant={summary['constant_feature_count']}  near-zero={summary['near_zero_counts']}")
    print(f"  |r| pair counts={summary['correlation']['pair_counts_abs_r']}  "
          f"max |r|={summary['correlation']['max_abs_correlation']:.6f}")
    print(f"  conclusion: {summary['conclusion']['text']}")
    return summary


if __name__ == "__main__":
    main()
