
from pathlib import Path

import numpy as np
import pandas as pd

from feature_fusion import (
    load_aligned_features,
    get_feature_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features" / "export"

EXPECTED_SPLITS = {
    "train": 960,
    "val": 240,
    "test": 240,
}

METADATA_COLS = [
    "sample_id",
    "trial_id",
    "actor_id",
    "emotion_label",
    "split",
]


def main():
    # Load and align M1 handcrafted and M2 Transformer features
    merged, m1 = load_aligned_features()

    groups = get_feature_groups(merged, m1)

    # Preserve exact feature ordering used by fusion
    deep_cols = groups["transformer_only"]
    combined_cols = groups["handcrafted_transformer"]

    handcrafted_cols = [
        c for c in combined_cols
        if c not in deep_cols
    ]

    assert len(handcrafted_cols) == 48
    assert len(deep_cols) == 128
    assert len(combined_cols) == 176

    # Select metadata and features
    export_df = merged[
        [
            "sample_id_m1",
            "trial_id",
            "actor_id_m1",
            "emotion_label",
            "split_m1",
        ]
        + combined_cols
    ].copy()

    # Rename metadata and feature columns
    rename_map = {
        "sample_id_m1": "sample_id",
        "actor_id_m1": "actor_id",
        "split_m1": "split",
        **{
            c: f"m1_{c}"
            for c in handcrafted_cols
        },
        **{
            c: f"m2_{c}"
            for c in deep_cols
        },
    }

    export_df = export_df.rename(columns=rename_map)

    # Validate the complete export
    feature_cols = [
        c for c in export_df.columns
        if c not in METADATA_COLS
    ]

    assert export_df.shape == (1440, 181)
    assert len(feature_cols) == 176

    assert export_df["sample_id"].is_unique
    assert export_df["trial_id"].is_unique

    assert np.isfinite(
        export_df[feature_cols].to_numpy(dtype=np.float64)
    ).all()

    assert (
        export_df["split"].value_counts().to_dict()
        == EXPECTED_SPLITS
    )

    # Sort consistently by actor and recording
    export_df = export_df.sort_values(
        ["actor_id", "sample_id"]
    ).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Full export
    full_path = OUTPUT_DIR / "audio_fused_176d.csv"
    export_df.to_csv(full_path, index=False)

    print("Full export:", export_df.shape)
    print("Feature dimensions:", len(feature_cols))
    print("Saved:", full_path)

    # Partition-specific exports
    for split in ("train", "val", "test"):
        split_df = export_df[
            export_df["split"] == split
        ]

        output_path = (
            OUTPUT_DIR / f"audio_fused_176d_{split}.csv"
        )

        split_df.to_csv(output_path, index=False)

        print(
            f"{split:5s} | "
            f"Rows: {len(split_df):4d} | "
            f"Saved: {output_path}"
        )

    print("\nAudio feature export completed successfully!")


if __name__ == "__main__":
    main()
