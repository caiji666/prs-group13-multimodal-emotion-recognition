# Speech changes over time, but over short intervals (roughly 20–30 ms)
# Frame length we will take 25 ms --- 400 samples per Frame
# and we will take Hop length of 10ms -- 160 samples hop length
# Overlapping frames allow us to analyze more frequent measurements of the signal. 
# If frames don't overlap, the change might occur near the boundary between two frames.


import numpy as np
import librosa


from pathlib import Path
from src.preprocessing import preprocess_audio

SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 25
DEFAULT_HOP_MS = 10


def frame_audio(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    hop_ms: int = DEFAULT_HOP_MS,
) -> np.ndarray:
    """Divide audio into overlapping frames."""

    frame_length = round(sr * frame_ms / 1000)
    hop_length = round(sr * hop_ms / 1000)

    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Frame and hop lengths must be positive.")

    if audio.ndim != 1:
        raise ValueError("Expected mono audio.")

    if len(audio) < frame_length:
        raise ValueError("Audio is shorter than one frame.")

    frames = librosa.util.frame(
        audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return frames


def extract_zcr(frames: np.ndarray) -> np.ndarray:
    """Calculate the zero-crossing rate for each frame."""

    signs = np.signbit(frames)

    crossings = signs[1:, :] != signs[:-1, :]

    zcr = np.mean(crossings, axis=0)

    return zcr


def extract_rms(frames: np.ndarray) -> np.ndarray:
    """Calculate RMS amplitude for each frame."""

    rms = np.sqrt(
        np.mean(frames ** 2, axis=0)
    )

    return rms

def extract_energy(frames: np.ndarray) -> np.ndarray:
    """Calculate signal energy for each frame."""

    energy = np.sum(frames ** 2, axis=0)

    return energy

def summarize_feature(values: np.ndarray, prefix: str) -> dict:
    """Summarize frame-level features into recording-level statistics."""

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def extract_time_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = DEFAULT_FRAME_MS,
    hop_ms: int = DEFAULT_HOP_MS,
) -> dict:
    """Extract 12 recording-level time-domain features."""

    frames = frame_audio(
        audio,
        sr=sr,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )

    zcr = extract_zcr(frames)
    rms = extract_rms(frames)
    energy = extract_energy(frames)

    features = {}

    features.update(summarize_feature(zcr, "zcr"))
    features.update(summarize_feature(rms, "rms"))
    features.update(summarize_feature(energy, "energy"))

    return features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    # for frame_ms in (20, 25, 30):

    #     frames = frame_audio(
    #         audio,
    #         sr=sr,
    #         frame_ms=frame_ms,
    #         hop_ms=10,
    #     )

    #     frame_length = round(sr * frame_ms / 1000)
    #     hop_length = round(sr * 10 / 1000)

    #     print(
    #         f"Frame: {frame_ms} ms | "
    #         f"Frame samples: {frame_length} | "
    #         f"Hop samples: {hop_length} | "
    #         f"Overlap: {frame_length - hop_length} | "
    #         f"Shape: {frames.shape}"
    #     )

    frames = frame_audio(
        audio,
        sr=sr,
        frame_ms=25,
        hop_ms=10,
    )

    zcr = extract_zcr(frames)
    print("\nZCR shape:", zcr.shape)
    print("First 5 ZCR values:", zcr[:5])
    print("Mean ZCR:", np.mean(zcr))
    
    rms = extract_rms(frames)
    print("\nRMS shape:", rms.shape)
    print("First 5 RMS values:", rms[:5])
    print("Mean RMS:", np.mean(rms))

    energy = extract_energy(frames)

    print("\nEnergy shape:", energy.shape)
    print("First 5 Energy values:", energy[:5])
    print("Mean Energy:", np.mean(energy))

