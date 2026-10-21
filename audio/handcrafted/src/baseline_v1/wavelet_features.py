
from pathlib import Path

import numpy as np
import pywt

from src.preprocessing import preprocess_audio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAVELET = "db4"
DWT_LEVEL = 4


def extract_wavelet_features(
    audio: np.ndarray,
    wavelet: str = WAVELET,
    level: int = DWT_LEVEL,
) -> dict:
    """Extract fixed-length statistics from DWT coefficients."""

    audio = np.asarray(audio, dtype=np.float64)

    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must be nonempty and finite")

    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(
        len(audio),
        wavelet_obj.dec_len,
    )

    if level > max_level:
        raise ValueError(
            f"Requested DWT level {level} exceeds "
            f"maximum supported level {max_level}"
        )

    # Coefficients: [A4, D4, D3, D2, D1]
    coeffs = pywt.wavedec(
        audio,
        wavelet=wavelet,
        level=level,
        mode="symmetric",
    )

    names = [f"A{level}"] + [
        f"D{i}" for i in range(level, 0, -1)
    ]

    features = {}

    for name, coeff in zip(names, coeffs):
        coeff = np.asarray(coeff, dtype=np.float64)

        squared = coeff ** 2
        total_energy = squared.sum()

        # Normalized squared coefficients
        if total_energy > 0:
            probabilities = squared / total_energy
            nonzero = probabilities > 0
            entropy = -np.sum(
                probabilities[nonzero]
                * np.log2(probabilities[nonzero])
            )
        else:
            entropy = 0.0

        features[f"dwt_{name}_mean"] = float(np.mean(coeff))
        features[f"dwt_{name}_variance"] = float(np.var(coeff))
        features[f"dwt_{name}_energy"] = float(np.mean(squared))
        features[f"dwt_{name}_entropy"] = float(entropy)

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

    features = extract_wavelet_features(audio)

    print("Sample rate:", sr)
    print("Audio samples:", len(audio))
    print("Wavelet:", WAVELET)
    print("Decomposition level:", DWT_LEVEL)
    print("Extracted features:", len(features))

    for name, value in features.items():
        print(f"{name:22s}: {value:.6f}")

    assert len(features) == 20
    assert np.isfinite(list(features.values())).all()

    print("\nAll wavelet feature checks passed!")
