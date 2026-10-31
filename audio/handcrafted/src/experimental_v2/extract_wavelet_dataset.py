
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing import preprocess_audio
from src.experimental_v2.wavelet_features import (
    extract_wavelet_features,
    WAVELETS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METADATA_PATH = PROJECT_ROOT / "data/audio_splits.csv"
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/features/experimental_v2/dwt"
)

METADATA_COLUMNS = [
    "sample_id",
    "actor_id",
    "emotion_label",
    "split",
]


def extract_dwt_datasets(metadata, normalize):
    """Extract all five wavelet representations."""

    records = {wavelet: [] for wavelet in WAVELETS}

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

        for wavelet in WAVELETS:

            features = extract_wavelet_features(
                audio,
                wavelet=wavelet,
                level=4,
            )

            record = {
                "sample_id": row.sample_id,
                "actor_id": row.actor_id,
                "emotion_label": row.emotion_label,
                "split": row.split,
                **features,
            }

            records[wavelet].append(record)

    return {
        wavelet: pd.DataFrame(rows)
        for wavelet, rows in records.items()
    }


if __name__ == "__main__":

    metadata = pd.read_csv(METADATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for normalize in (False, True):

        norm_label = "norm" if normalize else "raw"

        print(f"\nExtracting DWT: {norm_label}")

        datasets = extract_dwt_datasets(
            metadata,
            normalize=normalize,
        )

        for wavelet, dataset in datasets.items():

            assert len(dataset) == 1440
            assert dataset["sample_id"].is_unique

            feature_columns = [
                col for col in dataset.columns
                if col not in METADATA_COLUMNS
            ]

            assert len(feature_columns) == 20
            assert np.isfinite(
                dataset[feature_columns].to_numpy()
            ).all()

            output_path = (
                OUTPUT_DIR / f"dwt_{wavelet}_{norm_label}.csv"
            )

            dataset.to_csv(output_path, index=False)

            print(
                f"{wavelet}: {dataset.shape} "
                f"-> {output_path.name}"
            )

    print("\nAll 10 DWT datasets generated!")
