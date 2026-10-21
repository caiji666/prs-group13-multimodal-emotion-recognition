
from pathlib import Path

import numpy as np
import librosa

from src.preprocessing import preprocess_audio


SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def extract_spectral_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_length: int = FRAME_LENGTH,
    hop_length: int = HOP_LENGTH,
) -> dict:
    """Extract frame-level FFT-based spectral features."""
    
    if len(audio) < frame_length:
        raise ValueError(
            f"Audio too short: {len(audio)} samples; "
            f"minimum required: {frame_length}"
        )
    
    # Divide audio into overlapping frames
    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float64)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(frame_length)[:, None]
    windowed_frames = frames * window

    # Real-valued FFT along the sample axis
    fft_values = np.fft.rfft(
        windowed_frames,
        axis=0,
    )

    magnitude = np.abs(fft_values)
    power = magnitude ** 2

    # Frequency corresponding to each FFT bin
    frequencies = np.fft.rfftfreq(
        frame_length,
        d=1 / sr,
    )

    # Avoid division by zero
    eps = 1e-10
    magnitude_sum = magnitude.sum(axis=0) + eps

    # Spectral centroid (Hz)
    centroid = (
        frequencies[:, None] * magnitude
    ).sum(axis=0) / magnitude_sum

    # Spectral bandwidth (Hz)
    bandwidth = np.sqrt(
        (
            (frequencies[:, None] - centroid[None, :]) ** 2
            * magnitude
        ).sum(axis=0) / magnitude_sum
    )

    # Spectral rolloff (85% cumulative power)
    cumulative_power = np.cumsum(power, axis=0)
    threshold = 0.85 * power.sum(axis=0)

    rolloff_indices = np.argmax(
        cumulative_power >= threshold[None, :],
        axis=0,
    )

    rolloff = frequencies[rolloff_indices]

    # Spectral energy (windowed FFT power)
    spectral_energy = power.sum(axis=0)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_energy": spectral_energy,
    }



if __name__ == "__main__":

    sample_file = (
        PROJECT_ROOT
        / "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    )

    audio, sr = preprocess_audio(
        sample_file,
        normalize=False,
        trim=True,
    )

    features = extract_spectral_features(audio, sr=sr)

    print("Audio shape:", audio.shape)

    for name, values in features.items():
        print(
            f"{name:20s} | "
            f"Shape: {values.shape} | "
            f"Mean: {values.mean():.4f} | "
            f"Std: {values.std():.4f}"
        )

    # Validate all feature values
    expected_frames = 1 + (
        (len(audio) - FRAME_LENGTH) // HOP_LENGTH
    )

    for name, values in features.items():
        assert np.isfinite(values).all(), (
            f"Invalid values detected in {name}"
        )

        assert len(values) == expected_frames, (
            f"Unexpected frame count for {name}"
        )

    # Validate frequency-based features
    nyquist = sr / 2

    for name in (
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
    ):
        values = features[name]

        assert np.all(values >= 0), (
            f"Negative values in {name}"
        )

        assert np.all(values <= nyquist + 1e-6), (
            f"Values exceed Nyquist frequency in {name}"
        )

    # Spectral energy must be nonnegative
    assert np.all(features["spectral_energy"] >= 0)

    print("\nAll spectral feature validation checks passed!")
