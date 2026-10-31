
import numpy as np
import pandas as pd

from src.fusion.m2_sources import (
    PROJECT_ROOT,
    HANDCRAFTED_PATH,
    M2_SOURCES,
    feature_dir,
    parse_source,
)


M1_PATH = (
    PROJECT_ROOT
    / "outputs/features/experimental_v2/combined/db2"
    / "combined_norm_20ms.csv"
)

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def main():

    source = parse_source(
        "Align M1 handcrafted and M2 learned audio features."
    )

    config = M2_SOURCES[source]
    m2_dir = config["input_dir"]
    output_dir = feature_dir(source)

    print("M2 source:", source)
    print("M2 input:", m2_dir)

    # 1. Load M1 handcrafted dataset
    m1 = pd.read_csv(M1_PATH)

    assert len(m1) == 1440
    assert m1["sample_id"].is_unique

    m1_features = [
        col for col in m1.columns
        if col not in METADATA_COLUMNS
    ]

    assert len(m1_features) == 48

    # M1 sample_id has 7 filename components.
    # Remove the modality prefix (03).
    assert m1["sample_id"].str.startswith("03-").all()

    m1["trial_id"] = m1["sample_id"].str.split(
        "-", n=1
    ).str[1]

    # 2. Load M2 learned feature datasets
    m2 = pd.concat(
        [
            pd.read_csv(m2_dir / f"{split}.csv")
            for split in ["train", "val", "test"]
        ],
        ignore_index=True,
    )

    assert len(m2) == 1440
    assert m2["sample_id"].is_unique

    # Exports name their 128 pooled features differently.
    # Any other columns (e.g. WavLM P_* probabilities) are ignored.
    source_features = [
        f"{config['feature_prefix']}{i:03d}"
        for i in range(128)
    ]

    assert set(source_features).issubset(m2.columns)

    m2_features = [
        f"f_{i:03d}"
        for i in range(128)
    ]

    # M2 sample_id corresponds to M1 trial_id.
    m2 = m2.rename(
        columns={
            "sample_id": "trial_id",
            "actor_id": "m2_actor_id",
            "split": "m2_split",
            "label": "m2_label",
            **dict(zip(source_features, m2_features)),
        }
    )

    # 3. Merge by recording identifier
    merged = m1.merge(
        m2[
            [
                "trial_id",
                "m2_actor_id",
                "m2_split",
                "m2_label",
                *m2_features,
            ]
        ],
        on="trial_id",
        how="inner",
        validate="one_to_one",
    )

    assert len(merged) == 1440

    # 4. Validate actor and split alignment
    assert (
        merged["actor_id"].to_numpy()
        == merged["m2_actor_id"].to_numpy()
    ).all()

    assert (
        merged["split"].to_numpy()
        == merged["m2_split"].to_numpy()
    ).all()

    # 5. Validate emotion label alignment
    label_map = {
        emotion: index
        for index, emotion in enumerate(EMOTIONS)
    }

    expected_labels = merged["emotion_label"].map(
        label_map
    )

    assert expected_labels.notna().all()

    assert (
        expected_labels.to_numpy()
        == merged["m2_label"].to_numpy()
    ).all()

    # 6. Rename features to prevent collisions
    handcrafted = merged[
        ["sample_id", "trial_id", "actor_id",
         "emotion_label", "split", *m1_features]
    ].copy()

    handcrafted = handcrafted.rename(
        columns={
            col: f"m1_{col}"
            for col in m1_features
        }
    )

    learned = merged[
        ["sample_id", "trial_id", "actor_id",
         "emotion_label", "split", *m2_features]
    ].copy()

    learned = learned.rename(
        columns={
            col: f"m2_{col}"
            for col in m2_features
        }
    )

    combined = handcrafted.merge(
        learned,
        on=[
            "sample_id",
            "trial_id",
            "actor_id",
            "emotion_label",
            "split",
        ],
        validate="one_to_one",
    )

    # 7. Final validation
    assert handcrafted.shape == (1440, 53)
    assert learned.shape == (1440, 133)
    assert combined.shape == (1440, 181)

    for df in [handcrafted, learned, combined]:
        feature_cols = [
            col for col in df.columns
            if col not in [
                "sample_id",
                "trial_id",
                "actor_id",
                "emotion_label",
                "split",
            ]
        ]

        assert np.isfinite(
            df[feature_cols].to_numpy()
        ).all()

    # 8. Save aligned datasets
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        HANDCRAFTED_PATH: handcrafted,
        output_dir / "learned_128d.csv": learned,
        output_dir / "combined_176d.csv": combined,
    }

    for output_path, df in outputs.items():
        df.to_csv(output_path, index=False)

        print(
            f"{output_path.name}: "
            f"{df.shape[0]} recordings, "
            f"{df.shape[1] - 5} features"
        )

    print("\nAll M1–M2 alignment checks passed!")
    print("Saved to:", output_dir)


if __name__ == "__main__":
    main()
