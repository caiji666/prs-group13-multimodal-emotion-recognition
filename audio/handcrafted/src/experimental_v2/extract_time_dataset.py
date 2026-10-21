
from pathlib import Path

import pandas as pd
import numpy as np

from src.preprocessing import preprocess_audio
from src.experimental_v2.time_features import extract_time_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = PROJECT_ROOT / "data/audio_splits.csv"
FEATURE_DIR = (
    PROJECT_ROOT / "outputs/features/experimental_v2"
)
OUTPUT_DIR = FEATURE_DIR / "time"


def extract_dataset(metadata, normalize, frame_ms):

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

        features = extract_time_features(
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

            assert len(feature_columns) == 12
            assert np.isfinite(
                dataset[feature_columns].to_numpy()
            ).all()

            norm_label = "norm" if normalize else "raw"

            output_path = (
                OUTPUT_DIR
                / f"time_{norm_label}_{frame_ms}ms.csv"
            )

            dataset.to_csv(
                output_path,
                index=False,
            )

            print("Saved:", output_path)
            print("Shape:", dataset.shape)
