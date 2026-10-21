
from pathlib import Path

import numpy as np
import pywt

from src.preprocessing import preprocess_audio
from src.experimental_v2.time_features import summarize_feature

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAVELETS = ("haar", "db2", "db4", "sym4", "db8")
DEFAULT_WAVELET = "db4"
LEVEL = 4


def compute_dwt(
    audio: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = LEVEL,
) -> list[np.ndarray]:
    """Perform multilevel DWT decomposition."""

    coefficients = pywt.wavedec(
        audio,
        wavelet=wavelet,
        level=level,
    )

    return coefficients


def extract_wavelet_features(
    audio: np.ndarray,
    wavelet: str = DEFAULT_WAVELET,
    level: int = LEVEL,
) -> dict:
    """Extract 20 recording-level DWT features."""

    coefficients = compute_dwt(
        audio,
        wavelet=wavelet,
        level=level,
    )

    names = ["a4", "d4", "d3", "d2", "d1"]

    features = {}

    for name, values in zip(names, coefficients):
        features.update(
            summarize_feature(values, name)
        )

    return features


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

    for wavelet in WAVELETS:

        features = extract_wavelet_features(
            audio,
            wavelet=wavelet,
            level=LEVEL,
        )

        print(f"\nWavelet: {wavelet}")
        print("Total features:", len(features))

        for name, value in features.items():
            print(f"{name}: {value:.6f}")
