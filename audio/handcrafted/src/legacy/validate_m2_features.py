
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

M1_PATH = (
    PROJECT_ROOT
    / "outputs/features/handcrafted_features.csv"
)

M2_DIR = PROJECT_ROOT / "data/external/m2_audio/pooled"

LABEL_MAP = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprise": 7,
}


def main():
    # Load M1 handcrafted features
    m1 = pd.read_csv(M1_PATH)

    # Load M2 pooled Transformer features
    m2 = pd.concat(
        [
            pd.read_csv(M2_DIR / f"{split}.csv")
            for split in ("train", "val", "test")
        ],
        ignore_index=True,
    )

    print("M1 recordings:", len(m1))
    print("M2 recordings:", len(m2))

    # Convert M1 IDs to M2's format
    m1["m2_sample_id"] = (
        m1["sample_id"]
        .str.split("-")
        .str[1:]
        .str.join("-")
    )

    m1["label"] = m1["emotion_label"].map(LABEL_MAP)

    if m1["label"].isna().any():
        raise ValueError("Unknown M1 emotion labels")

    # Check identifiers before merging
    assert m1["m2_sample_id"].is_unique
    assert m2["sample_id"].is_unique

    m2_feature_cols = [
        f"f_{i:03d}" for i in range(128)
    ]

    assert set(m2_feature_cols).issubset(m2.columns)
    m1 = m1.copy()
    m2 = m2.copy()
    # Merge strictly by recording ID
    merged = m1.merge(
        m2,
        left_on="m2_sample_id",
        right_on="sample_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_m1", "_m2"),
    )

    print("\nRecording matching:")
    print(merged["_merge"].value_counts())

    assert (merged["_merge"] == "both").all(), (
        "Some recordings do not match"
    )

    # Verify actor, partition and emotion
    assert (
        merged["actor_id_m1"]
        == merged["actor_id_m2"]
    ).all(), "Actor mismatch"

    assert (
        merged["split_m1"]
        == merged["split_m2"]
    ).all(), "Split mismatch"

    assert (
        merged["label_m1"]
        == merged["label_m2"]
    ).all(), "Emotion label mismatch"

    # Verify learned features
    deep_features = merged[
        m2_feature_cols
    ].to_numpy(dtype=np.float64)

    assert deep_features.shape == (1440, 128)
    assert np.isfinite(deep_features).all()

    print("\nM2 feature matrix:", deep_features.shape)

    print("\nSplit distribution:")
    print(merged["split_m1"].value_counts())

    print("\nAll M1/M2 alignment checks passed!")


if __name__ == "__main__":
    main()
