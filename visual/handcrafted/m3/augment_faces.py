"""Clip-consistent, train-only augmentation of the shared 224x224 face clips (M3 DWT-SVM only).

Each augmented clip gets ONE parameter set, applied identically to all 16
frames (no per-frame jitter). Parameters depend only on
(config name, source sample_id, global seed) through a SHA-256 hash, so they do
not change with the number of workers, the file order or between runs.

    AUG-0  no augmentation (baseline)
    AUG-1  horizontal flip of the whole clip
    AUG-2  light combined: flip (p=0.5), rotation, translation, scale,
           brightness and contrast within AUG2_RANGES

The input array is never modified; the output is a new uint8 [16, 224, 224, 3]
array. Geometric transforms use cv2.BORDER_REFLECT_101, so no black borders.

The augmented clips are used only to train the handcrafted DWT-SVM. They have
no matching M4 deep features, so they cannot be used for Handcrafted + Deep
per-sample concatenation unless M4 exports features for the same clips.
"""
import hashlib

import cv2
import numpy as np

from m3.dwt_features import FRAME_SIZE, N_FRAMES

AUG_CONFIGS = ("AUG-0", "AUG-1", "AUG-2")  # also the tie-breaking order (simplest first)
TRAINING_AUGMENTATION = {"AUG-0": "none", "AUG-1": "horizontal_flip",
                         "AUG-2": "light_combined"}
AUG2_RANGES = {
    "flip_probability": 0.5,
    "rotation_deg": (-5.0, 5.0),
    "translate_x": (-0.03, 0.03),  # fraction of the frame width
    "translate_y": (-0.03, 0.03),  # fraction of the frame height
    "scale": (0.97, 1.03),
    "brightness": (0.90, 1.10),
    "contrast": (0.90, 1.10),
}
PARAM_KEYS = ("flip", "rotation_deg", "translate_x", "translate_y", "scale", "brightness",
              "contrast")
IDENTITY = {"flip": False, "rotation_deg": 0.0, "translate_x": 0.0, "translate_y": 0.0,
            "scale": 1.0, "brightness": 1.0, "contrast": 1.0}


def local_seed(config_name, sample_id, seed):
    """Stable per-clip seed (SHA-256, not Python's per-process hash())."""
    key = f"{config_name}|{sample_id}|{int(seed)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def sample_params(config_name, sample_id, seed):
    """The single parameter set for one clip."""
    if config_name == "AUG-0":
        return dict(IDENTITY)
    if config_name == "AUG-1":
        return {**IDENTITY, "flip": True}
    if config_name == "AUG-2":
        rng = np.random.default_rng(local_seed(config_name, sample_id, seed))
        r = AUG2_RANGES
        return {  # fixed draw order, so parameters are reproducible
            "flip": bool(rng.random() < r["flip_probability"]),
            "rotation_deg": float(rng.uniform(*r["rotation_deg"])),
            "translate_x": float(rng.uniform(*r["translate_x"])),
            "translate_y": float(rng.uniform(*r["translate_y"])),
            "scale": float(rng.uniform(*r["scale"])),
            "brightness": float(rng.uniform(*r["brightness"])),
            "contrast": float(rng.uniform(*r["contrast"])),
        }
    raise ValueError(f"Unknown augmentation config {config_name!r}")


def check_params(params):
    """Raise ValueError if any parameter lies outside the AUG-2 ranges."""
    if set(params) != set(PARAM_KEYS):
        raise ValueError(f"Expected parameters {PARAM_KEYS}, got {sorted(params)}")
    for k in PARAM_KEYS[1:]:
        lo, hi = AUG2_RANGES[k]
        if not lo <= params[k] <= hi:
            raise ValueError(f"{k}={params[k]} outside [{lo}, {hi}]")


def check_clip(frames):
    if frames.shape != (N_FRAMES, FRAME_SIZE, FRAME_SIZE, 3) or frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 [{N_FRAMES}, {FRAME_SIZE}, {FRAME_SIZE}, 3], "
                         f"got {frames.dtype} {frames.shape}")


def affine_matrix(params, size=FRAME_SIZE):
    """Rotation + scale about the centre, then translation (same matrix for every frame)."""
    c = (size - 1) / 2
    M = cv2.getRotationMatrix2D((c, c), params["rotation_deg"], params["scale"])
    M[0, 2] += params["translate_x"] * size
    M[1, 2] += params["translate_y"] * size
    return M


def augment_clip(frames, params):
    """Apply one parameter set to all frames of a clip; returns a new uint8 array."""
    check_clip(frames)
    check_params(params)
    out = frames[:, :, ::-1] if params["flip"] else frames  # horizontal flip (width axis)
    geometric = (params["rotation_deg"], params["translate_x"], params["translate_y"],
                 params["scale"]) != (0.0, 0.0, 0.0, 1.0)
    if geometric:
        M = affine_matrix(params)
        out = np.stack([cv2.warpAffine(np.ascontiguousarray(f), M, (FRAME_SIZE, FRAME_SIZE),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT_101) for f in out])
    if params["brightness"] != 1.0 or params["contrast"] != 1.0:
        x = out.astype(np.float32)
        pivot = x.mean()  # one pivot for the whole clip, so all frames get the same mapping
        x = ((x - pivot) * params["contrast"] + pivot) * params["brightness"]
        out = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    out = np.ascontiguousarray(out, dtype=np.uint8)
    if out is frames or np.shares_memory(out, frames):
        out = out.copy()
    check_clip(out)
    return out
