
from pathlib import Path

import numpy as np
import librosa

from preprocessing import preprocess_audio


# Configuration
SAMPLE_RATE = 16000
FRAME_LENGTH = 400  # 25 ms
HOP_LENGTH = 160     # 10 ms

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_time_features(
    audio: np.ndarray,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> dict:
    """Extract frame-level ZCR, RMS and Energy."""

    # Divide audio into overlapping frames
    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # Convert to float64 for numerical precision
    frames_float = frames.astype(np.float64)

    # Zero-Crossing Rate
    zcr = np.mean(
        np.signbit(frames[1:, :])
        != np.signbit(frames[:-1, :]),
        axis=0,
    )

    # Root Mean Square
    rms = np.sqrt(
        np.mean(frames_float ** 2, axis=0)
    )

    # Frame Energy
    energy = np.sum(
        frames_float ** 2,
        axis=0,
    )

    return {
        "zcr": zcr,
        "rms": rms,
        "energy": energy,
    }


if __name__ == "__main__":

    # Test on one RAVDESS recording
    sample_file = (
        PROJECT_ROOT
        / "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    )

    audio, sr = preprocess_audio(
        sample_file,
        normalize=False,
        trim=True,
    )

    # Extract frame-level features
    features = extract_time_features(audio)

    print("Audio shape:", audio.shape)
    print("Sample rate:", sr)
    print("Frame duration:", FRAME_LENGTH / sr, "seconds")
    print("Hop duration:", HOP_LENGTH / sr, "seconds")

    print("\nExtracted features:")

    for name, values in features.items():
        print(
            f"{name.upper():8s} | "
            f"Shape: {values.shape} | "
            f"Mean: {values.mean():.6f} | "
            f"Std: {values.std():.6f}"
        )

    # Verify Energy = Frame Length × RMS²
    np.testing.assert_allclose(
        features["energy"],
        FRAME_LENGTH * features["rms"] ** 2,
        rtol=1e-6,
    )

    print("\nEnergy-RMS consistency check passed!")
