
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw"
OUTPUT_PATH = PROJECT_ROOT / "data/metadata.csv"


METADATA_COLUMNS = [
    "sample_id",
    "file_path",
    "actor_id",
    "gender",
    "emotion_code",
    "emotion_label",
    "intensity",
    "statement",
    "repetition",
]


EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


EXPECTED_COUNTS = {
    "neutral": 96,
    "calm": 192,
    "happy": 192,
    "sad": 192,
    "angry": 192,
    "fearful": 192,
    "disgust": 192,
    "surprised": 192,
}


def parse_filename(audio_file: Path) -> dict:
    """Extract and validate metadata from a RAVDESS filename."""

    sample_id = audio_file.stem
    parts = sample_id.split("-")

    if audio_file.suffix.lower() != ".wav" or len(parts) != 7:
        raise ValueError(f"Invalid filename: {audio_file.name}")

    (
        modality,
        vocal_channel,
        emotion_code,
        intensity,
        statement,
        repetition,
        actor,
    ) = parts

    # Validate RAVDESS identifiers
    if modality != "03":
        raise ValueError(f"Expected audio-only recording: {sample_id}")

    if vocal_channel != "01":
        raise ValueError(f"Expected speech recording: {sample_id}")

    if emotion_code not in EMOTION_MAP:
        raise ValueError(f"Invalid emotion code: {sample_id}")

    if intensity not in ("01", "02"):
        raise ValueError(f"Invalid intensity: {sample_id}")

    if emotion_code == "01" and intensity != "01":
        raise ValueError(f"Neutral cannot have strong intensity: {sample_id}")

    if statement not in ("01", "02"):
        raise ValueError(f"Invalid statement: {sample_id}")

    if repetition not in ("01", "02"):
        raise ValueError(f"Invalid repetition: {sample_id}")

    if actor not in {f"{i:02d}" for i in range(1, 25)}:
        raise ValueError(f"Invalid actor ID: {sample_id}")

    actor_id = int(actor)

    # Verify the file belongs to its actor directory
    expected_actor_dir = f"Actor_{actor_id:02d}"

    if audio_file.parent.name != expected_actor_dir:
        raise ValueError(
            f"Actor directory mismatch: {audio_file}"
        )

    gender = "female" if actor_id % 2 == 0 else "male"

    return {
        "sample_id": sample_id,
        "file_path": audio_file.relative_to(PROJECT_ROOT).as_posix(),
        "actor_id": actor_id,
        "gender": gender,
        "emotion_code": int(emotion_code),
        "emotion_label": EMOTION_MAP[emotion_code],
        "intensity": int(intensity),
        "statement": int(statement),
        "repetition": int(repetition),
    }


def build_metadata() -> pd.DataFrame:
    """Build metadata for the RAVDESS audio-only speech subset."""

    audio_files = sorted(
        RAW_DIR.glob("Actor_*/*.wav")
    )

    if len(audio_files) != 1440:
        raise ValueError(
            f"Expected 1440 recordings, found {len(audio_files)}"
        )

    records = [
        parse_filename(audio_file)
        for audio_file in audio_files
    ]

    metadata = pd.DataFrame(
        records,
        columns=METADATA_COLUMNS,
    )

    # Validate metadata
    if not metadata["sample_id"].is_unique:
        raise ValueError("Duplicate sample IDs found.")

    if metadata["emotion_label"].value_counts().to_dict() != EXPECTED_COUNTS:
        raise ValueError("Unexpected emotion distribution.")

    actor_counts = metadata["actor_id"].value_counts()

    if len(actor_counts) != 24 or not (actor_counts == 60).all():
        raise ValueError("Unexpected actor distribution.")

    return metadata


def main():

    metadata = build_metadata()

    print("Metadata shape:", metadata.shape)
    print("\nFirst five recordings:")
    print(metadata.head())

    print("\nEmotion distribution:")
    print(metadata["emotion_label"].value_counts())

    print("\nRecordings per actor:")
    print(metadata["actor_id"].value_counts().sort_index())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
