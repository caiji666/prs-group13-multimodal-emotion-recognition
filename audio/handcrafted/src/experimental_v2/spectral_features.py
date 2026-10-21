
from pathlib import Path

from src.experimental_v2.time_features import (
    frame_audio,
    summarize_feature,
)

import numpy as np
import librosa

from src.preprocessing import preprocess_audio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 25
DEFAULT_HOP_MS = 10


def compute_spectrum(
    frames: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude spectrum for each audio frame."""

    # Apply Hann window to each frame
    window = np.hanning(frames.shape[0])

    windowed_frames = frames * window[:, np.newaxis]

    # Real-valued FFT along the sample axis
    spectrum = np.fft.rfft(
        windowed_frames,
        axis=0,
    )

    # Magnitude spectrum
    magnitudes = np.abs(spectrum)

    # Corresponding frequency bins
    frequencies = np.fft.rfftfreq(
        frames.shape[0],
        d=1 / sr,
    )

    return magnitudes, frequencies


def extract_spectral_centroid(
    magnitudes: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Calculate spectral centroid for each frame."""

    weighted_sum = np.sum(
        magnitudes * frequencies[:, np.newaxis],
        axis=0,
    )

    magnitude_sum = np.sum(
        magnitudes,
        axis=0,
    )

    centroid = weighted_sum / np.maximum(
        magnitude_sum,
        1e-10,
    )

    return centroid

def extract_spectral_bandwidth(
    magnitudes: np.ndarray,
    frequencies: np.ndarray,
    centroid: np.ndarray,
) -> np.ndarray:
    """Calculate spectral bandwidth for each frame."""

    squared_distance = (
        frequencies[:, np.newaxis] - centroid[np.newaxis, :]
    ) ** 2

    weighted_sum = np.sum(
        magnitudes * squared_distance,
        axis=0,
    )

    magnitude_sum = np.sum(
        magnitudes,
        axis=0,
    )

    bandwidth = np.sqrt(
        weighted_sum / np.maximum(magnitude_sum, 1e-10)
    )

    return bandwidth

def extract_spectral_rolloff(
    magnitudes: np.ndarray,
    frequencies: np.ndarray,
    roll_percent: float = 0.85,
) -> np.ndarray:
    """Calculate spectral rolloff for each frame."""

    power = magnitudes ** 2

    cumulative_energy = np.cumsum(
        power,
        axis=0,
    )

    total_energy = cumulative_energy[-1, :]

    threshold = roll_percent * total_energy

    rolloff_indices = np.argmax(
        cumulative_energy >= threshold[np.newaxis, :],
        axis=0,
    )

    rolloff = frequencies[rolloff_indices]

    return rolloff

def extract_spectral_flatness(
    magnitudes: np.ndarray,
) -> np.ndarray:
    """Calculate spectral flatness for each frame."""

    power = magnitudes ** 2

    # Avoid log(0)
    power = np.maximum(power, 1e-10)

    geometric_mean = np.exp(
        np.mean(np.log(power), axis=0)
    )

    arithmetic_mean = np.mean(
        power,
        axis=0,
    )

    flatness = geometric_mean / arithmetic_mean

    return flatness


def extract_spectral_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    hop_ms: int = DEFAULT_HOP_MS,
) -> dict:
    """Extract 16 recording-level spectral features."""

    # 1. Divide audio into overlapping frames
    frames = frame_audio(
        audio,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    # 2. Calculate magnitude spectrum
    magnitudes, frequencies = compute_spectrum(
        frames,
        sr,
    )

    # 3. Extract frame-level spectral features
    centroid = extract_spectral_centroid(
        magnitudes,
        frequencies,
    )

    bandwidth = extract_spectral_bandwidth(
        magnitudes,
        frequencies,
        centroid,
    )

    rolloff = extract_spectral_rolloff(
        magnitudes,
        frequencies,
        roll_percent=0.85,
    )

    flatness = extract_spectral_flatness(
        magnitudes,
    )

    # 4. Statistical pooling
    features = {}

    features.update(summarize_feature(centroid, "centroid"))
    features.update(summarize_feature(bandwidth, "bandwidth"))
    features.update(summarize_feature(rolloff, "rolloff"))
    features.update(summarize_feature(flatness, "flatness"))

    return features

# if __name__ == "__main__":

#     sample_file = (
#         PROJECT_ROOT
#         / "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
#     )

#     audio, sr = preprocess_audio(
#         sample_file,
#         normalize=False,
#         trim=True,
#     )

#     frames = frame_audio(
#         audio,
#         sr=sr,
#         frame_ms=25,
#         hop_ms=10,
#     )

#     magnitudes, frequencies = compute_spectrum(
#         frames,
#         sr,
#     )

#     print("Frames shape:", frames.shape)
#     print("Magnitudes shape:", magnitudes.shape)
#     print("Frequencies shape:", frequencies.shape)

#     print("First 10 frequency bins:", frequencies[:10])
#     print("Maximum frequency:", frequencies[-1])

#     centroid = extract_spectral_centroid(
#         magnitudes,
#         frequencies,
#     )

#     print("\nCentroid shape:", centroid.shape)
#     print("First 5 centroid values:", centroid[:5])
#     print("Mean centroid:", np.mean(centroid))

#     bandwidth = extract_spectral_bandwidth(
#         magnitudes,
#         frequencies,
#         centroid,
#     )

#     print("\nBandwidth shape:", bandwidth.shape)
#     print("First 5 bandwidth values:", bandwidth[:5])
#     print("Mean bandwidth:", np.mean(bandwidth))

#     rolloff = extract_spectral_rolloff(
#         magnitudes,
#         frequencies,
#         roll_percent=0.85,
#     )

#     print("\nRolloff shape:", rolloff.shape)
#     print("First 5 rolloff values:", rolloff[:5])
#     print("Mean rolloff:", np.mean(rolloff))

#     flatness = extract_spectral_flatness(
#         magnitudes
#     )

#     print("\nFlatness shape:", flatness.shape)
#     print("First 5 flatness values:", flatness[:5])
#     print("Mean flatness:", np.mean(flatness))


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

    spectral_features = extract_spectral_features(
        audio,
        sr=sr,
        frame_ms=25,
        hop_ms=10,
    )

    for name, value in spectral_features.items():
        print(f"{name}: {value:.6f}")

    print("\nTotal spectral features:", len(spectral_features))