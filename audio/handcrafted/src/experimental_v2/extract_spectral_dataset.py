
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_audio
from src.experimental_v2.spectral_features import extract_spectral_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data/audio_splits.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/features/experimental_v2/spectral"
)


def extract_dataset(metadata, normalize, frame_ms):
    """Extract spectral features for all recordings."""

    records = []

    for row in metadata.itertuples(index=False):

        file_path = (
            PROJECT_ROOT
            / "data/raw"
            / f"Actor_{row.actor_id:02d}"
            / f"{row.sample_id}.wav"
        )

        audio, sr = preprocess_audio(
            file_path,
            normalize=normalize,
            trim=True,
        )

        features = extract_spectral_features(
            audio,
            sr=sr,
            frame_ms=frame_ms,
            hop_ms=10,
        )

        record = {
            "sample_id": row.sample_id,
            "actor_id": row.actor_id,
            "emotion_label": row.emotion_label,
            "split": row.split,
            **features,
        }

        records.append(record)

    return pd.DataFrame(records)


if __name__ == "__main__":

    metadata = pd.read_csv(METADATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for normalize in (False, True):

        for frame_ms in (20, 25, 30):

            norm_label = "norm" if normalize else "raw"

            print(
                f"Extracting: normalize={normalize}, "
                f"frame={frame_ms}ms"
            )

            dataset = extract_dataset(
                metadata,
                normalize=normalize,
                frame_ms=frame_ms,
            )

            # Basic checks
            assert len(dataset) == 1440
            assert dataset["sample_id"].is_unique

            feature_columns = [
                col for col in dataset.columns
                if col not in (
                    "sample_id",
                    "actor_id",
                    "emotion_label",
                    "split",
                )
            ]

            assert len(feature_columns) == 16
            assert np.isfinite(
                dataset[feature_columns].to_numpy()
            ).all()

            output_path = (
                OUTPUT_DIR
                / f"spectral_{norm_label}_{frame_ms}ms.csv"
            )

            dataset.to_csv(
                output_path,
                index=False,
            )

            print("Saved:", output_path)
            print("Shape:", dataset.shape)

    
    # Check alignment between time and spectral datasets
    print("\nChecking time and spectral dataset alignment...")

    metadata_cols = [
        "actor_id",
        "emotion_label",
        "split",
    ]

    for normalize in (False, True):
        for frame_ms in (20, 25, 30):

            norm_label = "norm" if normalize else "raw"

            time_df = pd.read_csv(
                PROJECT_ROOT
                / "outputs/features/experimental_v2/time"
                / f"time_{norm_label}_{frame_ms}ms.csv"
            )

            spectral_df = pd.read_csv(
                OUTPUT_DIR
                / f"spectral_{norm_label}_{frame_ms}ms.csv"
            )

            time_metadata = (
                time_df
                .set_index("sample_id")[metadata_cols]
                .sort_index()
            )

            spectral_metadata = (
                spectral_df
                .set_index("sample_id")[metadata_cols]
                .sort_index()
            )

            assert time_df["sample_id"].is_unique
            assert spectral_df["sample_id"].is_unique

            assert time_metadata.equals(spectral_metadata), (
                f"Dataset mismatch: {norm_label}_{frame_ms}ms"
            )

            print(f"{norm_label}_{frame_ms}ms: Aligned!")

    print("\nAll time and spectral datasets are aligned!")


