
from pathlib import Path

import librosa
import numpy as np


TARGET_SR = 16000
TRIM_TOP_DB = 35


def load_audio(file_path: str | Path, sr: int = TARGET_SR):
    """Load a mono audio recording at the target sample rate."""

    audio, sample_rate = librosa.load(
        str(file_path),
        sr=sr,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError(f"Empty audio file: {file_path}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio values: {file_path}")

    return audio, sample_rate


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Peak-normalize audio without changing silent recordings."""

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    top_db: int = TRIM_TOP_DB,
) -> np.ndarray:
    """Remove leading and trailing low-energy segments."""

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
    )

    # Avoid returning an empty waveform.
    if trimmed.size == 0:
        return audio

    return trimmed


def preprocess_audio(
    file_path: str | Path,
    sr: int = TARGET_SR,
    normalize: bool = False,
    trim: bool = True,
):
    """Apply deterministic preprocessing to a WAV recording."""

    audio, sample_rate = load_audio(file_path, sr=sr)

    # Trim first, then normalize.
    if trim:
        audio = trim_silence(audio)

    if normalize:
        audio = normalize_audio(audio)

    return audio, sample_rate


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    sample_file = (
        project_root
        / "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    )

    original, original_sr = librosa.load(
        sample_file, sr=None, mono=True
    )

    processed, processed_sr = preprocess_audio(sample_file)

    print(f"Original sample rate: {original_sr}")
    print(f"Processed sample rate: {processed_sr}")

    print(f"Original duration: {len(original) / original_sr:.2f}s")
    print(f"Processed duration: {len(processed) / processed_sr:.2f}s")

    print(f"Original peak: {np.max(np.abs(original)):.4f}")
    print(f"Processed peak: {np.max(np.abs(processed)):.4f}")

    print(f"Processed samples: {len(processed)}")
