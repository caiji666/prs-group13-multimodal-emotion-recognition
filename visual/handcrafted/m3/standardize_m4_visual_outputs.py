"""Copy M4's deep visual NPZ into the Level-2 input format (field renames only).

    sample_id -> sample_id, split -> split, label -> label,
    class_names -> class_names (lower-cased), pooled_features -> F_visual_deep,
    sequence_features -> F_visual_sequence, probabilities -> P_visual_deep,
    source_model -> source_model, source_checkpoint -> source_checkpoint

Arrays keep their values, dtype, shape and row order; the input file is only read. The input is
checked (unique ids, 960/240/240, labels 0..7, finite values, probabilities, pooled = mean of the
16 frames) and, when the M3 export exists, matched to it BY sample_id (label and split).
An existing output is reused only if identical; otherwise the run stops unless --overwrite.

Usage:
    uv run python -m m3.standardize_m4_visual_outputs --input ../mobilenetv3_mean_features.npz --output external/m4_visual_deep_outputs.npz
"""
import argparse
import hashlib
import os
from pathlib import Path

import numpy as np

from m3.common import EMOTIONS

DEFAULT_M3 = "outputs_augmentation/m3_visual_handcrafted_augmented.npz"
MAPPING = {"sample_id": "sample_id", "split": "split", "label": "label", "class_names": "class_names",
           "pooled_features": "F_visual_deep", "sequence_features": "F_visual_sequence",
           "probabilities": "P_visual_deep", "source_model": "source_model",
           "source_checkpoint": "source_checkpoint"}
EXPECTED_COUNTS = {"train": 960, "val": 240, "test": 240}


class M4FormatError(ValueError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source(path, expected_counts=None):
    expected_counts = expected_counts or EXPECTED_COUNTS
    z = np.load(path, allow_pickle=False)
    missing = [k for k in MAPPING if k not in z.files]
    if missing:
        raise M4FormatError(f"{path}: missing fields {missing}; actual fields {sorted(z.files)}")
    d = {k: z[k] for k in MAPPING}
    n = len(d["sample_id"])
    if len(set(d["sample_id"].tolist())) != n:
        raise M4FormatError("duplicate sample_id")
    counts = {str(s): int(c) for s, c in zip(*np.unique(d["split"], return_counts=True))}
    if counts != expected_counts:
        raise M4FormatError(f"split counts {counts}, expected {expected_counts}")
    if d["label"].shape != (n,) or d["label"].min() < 0 or d["label"].max() > 7:
        raise M4FormatError("label must be [N] with values 0..7")
    S, F, P = d["sequence_features"], d["pooled_features"], d["probabilities"]
    if S.ndim != 3 or S.shape[:2] != (n, 16) or F.shape != (n, S.shape[2]) or P.shape != (n, 8):
        raise M4FormatError(f"shapes: sequence {S.shape}, pooled {F.shape}, probabilities {P.shape}")
    for name, X in (("sequence_features", S), ("pooled_features", F), ("probabilities", P)):
        if not np.isfinite(X).all():
            raise M4FormatError(f"{name} contains NaN/Inf")
    if P.min() < 0 or P.max() > 1 or np.abs(P.sum(axis=1) - 1).max() > 1e-4:
        raise M4FormatError("probabilities must lie in [0, 1] and each row must sum to 1")
    S64 = S.astype(np.float64)
    if (np.abs(S64.mean(axis=1) - F) > 1e-6 + 1e-5 * np.abs(S64).mean(axis=1)).any():
        raise M4FormatError("pooled_features is not the 16-frame mean of sequence_features")
    if [str(c).lower() for c in d["class_names"]] != EMOTIONS:
        raise M4FormatError(f"class_names {list(d['class_names'])} do not match {EMOTIONS}")
    return d


def check_against_m3(d, m3_path):
    """Same sample_id set as M3; label and split equal after aligning BY sample_id."""
    h = np.load(m3_path, allow_pickle=False)
    ids = d["sample_id"].tolist()
    if set(ids) != set(h["sample_id"].tolist()):
        raise M4FormatError("sample_id set differs from the M3 export")
    pos = {s: i for i, s in enumerate(ids)}
    order = np.array([pos[s] for s in h["sample_id"].tolist()])
    for k in ("label", "split"):
        bad = np.flatnonzero(d[k][order] != h[k])
        if bad.size:
            raise M4FormatError(f"{k} differs from M3 for sample_id {h['sample_id'][bad[0]]}")
    return {"rows_in_m3_order": bool((order == np.arange(len(order))).all())}


def standardize(d):
    out = {MAPPING[k]: v for k, v in d.items()}
    out["class_names"] = np.char.lower(d["class_names"])  # the only value change
    return out


def same_content(path, expected):
    z = np.load(path, allow_pickle=False)
    if set(z.files) != set(expected):
        return False
    return all(z[k].dtype == v.dtype and z[k].shape == v.shape and np.array_equal(z[k], v)
               for k, v in expected.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="external/m4_visual_deep_outputs.npz")
    ap.add_argument("--m3", default=DEFAULT_M3, help="M3 export to match by sample_id (skipped if missing)")
    ap.add_argument("--overwrite", action="store_true", help="replace a different existing output")
    args = ap.parse_args(argv)
    src, out = Path(args.input), Path(args.output)
    before = sha256(src)
    try:
        d = load_source(src)
        m3 = check_against_m3(d, args.m3) if Path(args.m3).exists() else None
    except M4FormatError as e:
        raise SystemExit(f"ERROR: {e}") from None
    expected = standardize(d)

    if out.exists() and same_content(out, expected):
        status = "reused (identical existing file)"
    else:
        if out.exists() and not args.overwrite:
            raise SystemExit(f"ERROR: {out} exists with different content; pass --overwrite to replace it")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez(tmp, **expected)
        os.replace(tmp, out)  # single file
        status = "written"
    if not same_content(out, expected):
        raise SystemExit(f"ERROR: {out} does not match the expected content")
    if sha256(src) != before:
        raise SystemExit("ERROR: the input file changed during conversion")

    print(f"Input  {src}  sha256 {before} (unchanged)")
    print(f"Output {out}: {status}")
    if m3 is not None:
        print(f"Matched to M3 by sample_id: label and split equal "
              f"(rows already in M3 order: {m3['rows_in_m3_order']})")
    z = np.load(out, allow_pickle=False)
    for k in z.files:
        v = z[k]
        print(f"  {k:18s} {str(v.dtype):8s} {v.shape}" + (f"  {v.item()!r}" if v.shape == () else ""))
    return {"status": status, "input_sha256": before}


if __name__ == "__main__":
    main()
