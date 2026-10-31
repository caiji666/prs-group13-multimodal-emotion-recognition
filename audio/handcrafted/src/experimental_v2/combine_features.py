
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "outputs/features/experimental_v2"
OUTPUT_DIR = FEATURE_DIR / "combined"

WAVELETS = ("haar", "db2", "db4", "sym4", "db8")
FRAME_LENGTHS = (20, 25, 30)

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def combine_features(
    wavelet: str,
    normalize: bool,
    frame_ms: int,
) -> pd.DataFrame:
    """Combine time, spectral and DWT features."""

    norm_label = "norm" if normalize else "raw"

    # 1. Load the three feature datasets
    time_df = pd.read_csv(
        FEATURE_DIR / "time"
        / f"time_{norm_label}_{frame_ms}ms.csv"
    )

    spectral_df = pd.read_csv(
        FEATURE_DIR / "spectral"
        / f"spectral_{norm_label}_{frame_ms}ms.csv"
    )

    dwt_df = pd.read_csv(
        FEATURE_DIR / "dwt"
        / f"dwt_{wavelet}_{norm_label}.csv"
    )

    # 2. Check that sample IDs are unique
    for df in (time_df, spectral_df, dwt_df):
        assert df["sample_id"].is_unique

    # 3. Verify metadata alignment
    metadata_cols = METADATA_COLUMNS[1:]

    reference = (
        time_df.set_index("sample_id")[metadata_cols]
        .sort_index()
    )

    for df in (spectral_df, dwt_df):
        current = (
            df.set_index("sample_id")[metadata_cols]
            .sort_index()
        )

        assert reference.equals(current), (
            "Metadata mismatch between feature datasets."
        )

    # 4. Retain metadata from time dataset only
    spectral_features = spectral_df.drop(
        columns=metadata_cols
    )

    dwt_features = dwt_df.drop(
        columns=metadata_cols
    )

    # 5. Merge all three datasets using sample_id
    combined = time_df.merge(
        spectral_features,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    combined = combined.merge(
        dwt_features,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    # 6. Basic checks
    assert len(combined) == 1440
    assert combined["sample_id"].is_unique

    feature_columns = [
        col for col in combined.columns
        if col not in METADATA_COLUMNS
    ]

    assert len(feature_columns) == 48

    assert np.isfinite(
        combined[feature_columns].to_numpy()
    ).all()

    return combined


if __name__ == "__main__":

    for wavelet in WAVELETS:

        wavelet_dir = OUTPUT_DIR / wavelet
        wavelet_dir.mkdir(parents=True, exist_ok=True)

        for normalize in (False, True):

            norm_label = "norm" if normalize else "raw"

            for frame_ms in FRAME_LENGTHS:

                combined = combine_features(
                    wavelet=wavelet,
                    normalize=normalize,
                    frame_ms=frame_ms,
                )

                output_path = (
                    wavelet_dir
                    / f"combined_{norm_label}_{frame_ms}ms.csv"
                )

                combined.to_csv(
                    output_path,
                    index=False,
                )

                print(
                    f"{wavelet} | {norm_label} | "
                    f"{frame_ms}ms | Shape: {combined.shape}"
                )

    print("\nAll 30 combined datasets generated!")
