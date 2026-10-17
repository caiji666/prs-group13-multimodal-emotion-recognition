
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import preprocess_audio
from time_features import extract_time_features
from spectral_features import extract_spectral_features
from wavelet_features import extract_wavelet_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features"


def aggregate_features(features: dict) -> dict:
    """Convert frame-level features into fixed-length statistics."""

    aggregated = {}

    for name, values in features.items():
        values = np.asarray(values, dtype=np.float64)

        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid feature values: {name}")

        aggregated[f"{name}_mean"] = float(np.mean(values))
        aggregated[f"{name}_std"] = float(np.std(values))
        aggregated[f"{name}_min"] = float(np.min(values))
        aggregated[f"{name}_max"] = float(np.max(values))

    return aggregated



def extract_recording_features(file_path: Path) -> dict:
    """Extract the complete handcrafted audio representation."""

    audio, sr = preprocess_audio(
        file_path,
        normalize=False,
        trim=True,
    )

    # Frame-level time-domain features
    time_features = extract_time_features(audio)

    # Frame-level spectral features
    spectral_features = extract_spectral_features(
        audio,
        sr=sr,
    )

    # Fixed-length DWT features
    wavelet_features = extract_wavelet_features(audio)

    # Aggregate frame-level features
    aggregated = aggregate_features({
        **time_features,
        **spectral_features,
    })

    # DWT statistics are already fixed-length
    duplicate_names = set(aggregated) & set(wavelet_features)

    if duplicate_names:
        raise ValueError(
            f"Duplicate feature names: {duplicate_names}"
        )

    return {
        **aggregated,
        **wavelet_features,
    }



def main():
    metadata = pd.read_csv(DATA_DIR / "audio_splits.csv")

    records = []

    for _, row in metadata.iterrows():
        file_path = PROJECT_ROOT / row["file_path"]

        features = extract_recording_features(file_path)

        records.append({
            "sample_id": row["sample_id"],
            "trial_id": row["trial_id"],
            "actor_id": row["actor_id"],
            "emotion_label": row["emotion_label"],
            "split": row["split"],
            **features,
        })

    feature_df = pd.DataFrame(records)

    # Validate output
    feature_columns = [
        col for col in feature_df.columns
        if col not in {
            "sample_id", "trial_id", "actor_id",
            "emotion_label", "split"
        }
    ]

    assert len(feature_df) == 1440
    assert len(feature_columns) == 48

    assert np.isfinite(
        feature_df[feature_columns].to_numpy()
    ).all()

    assert feature_df["sample_id"].is_unique
    assert feature_df["trial_id"].is_unique

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "handcrafted_features.csv"
    feature_df.to_csv(output_path, index=False)

    print("Recordings processed:", len(feature_df))
    print("Features per recording:", len(feature_columns))
    print("Feature matrix shape:", feature_df[feature_columns].shape)
    print("Output:", output_path)

    print("\nSample features:")
    print(feature_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
