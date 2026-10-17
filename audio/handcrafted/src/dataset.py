
from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprise",
}


def parse_filename(file_path: Path) -> dict:
    """Extract RAVDESS metadata from a WAV filename."""

    parts = file_path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(f"Invalid filename: {file_path.name}")

    modality, channel, emotion, intensity, statement, repetition, actor = parts

    if modality != "03" or channel != "01":
        raise ValueError(f"Not an audio-only speech file: {file_path.name}")

    if emotion not in EMOTION_MAP:
        raise ValueError(f"Unknown emotion: {file_path.name}")

    return {
        "sample_id": file_path.stem,
        "file_path": str(file_path.relative_to(PROJECT_ROOT)),
        "actor_id": int(actor),
        "emotion_code": emotion,
        "emotion_label": EMOTION_MAP[emotion],
        "intensity": int(intensity),
        "statement": int(statement),
        "repetition": int(repetition),
    }


def build_metadata(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Index all RAVDESS speech WAV recordings."""

    audio_files = sorted(raw_dir.glob("Actor_*/*.wav"))

    if not audio_files:
        raise FileNotFoundError(f"No WAV files found in {raw_dir}")

    records = [parse_filename(path) for path in audio_files]

    df = pd.DataFrame(records)

    if df["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample IDs detected")

    return df


def validate_metadata(df: pd.DataFrame) -> None:
    """Check dataset size, actors and emotion categories."""

    print(f"Total recordings: {len(df)}")
    print(f"Unique actors: {df['actor_id'].nunique()}")
    print(f"Emotion classes: {df['emotion_label'].nunique()}")

    print("\nEmotion distribution:")
    print(df["emotion_label"].value_counts().sort_index())

    print("\nRecordings per actor:")
    print(df.groupby("actor_id").size().to_string())

    assert len(df) == 1440, "Expected 1440 speech recordings"
    assert df["actor_id"].nunique() == 24, "Expected 24 actors"
    assert df["emotion_label"].nunique() == 8, "Expected 8 emotions"
    assert df.groupby("actor_id").size().eq(60).all(), (
        "Expected 60 recordings per actor"
    )


if __name__ == "__main__":
    metadata = build_metadata()
    validate_metadata(metadata)

    output_path = DATA_DIR / "metadata.csv"
    metadata.to_csv(output_path, index=False)

    print(f"\nMetadata saved to: {output_path}")
    print("\nFirst five records:")
    print(metadata.head().to_string(index=False))
