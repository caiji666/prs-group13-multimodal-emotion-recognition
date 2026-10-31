from pathlib import Path
import numpy as np
import librosa
from src.experimental_v2.validation import (
    validate_resampling,
    validate_trimming,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP_DB = 35
TARGET_SR = 16000


def preprocess_audio(
    file_path: str | Path,
    normalize: bool = False,
    trim: bool = True,
) -> tuple[np.ndarray, int]:
    """Load, resample, trim and optionally normalize audio."""

    # 1. Load original audio
    audio, original_sr = librosa.load(
        file_path,
        sr=None,
        mono=True,
    )

    # 2. Resample to 16 kHz
    audio = librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=TARGET_SR,
    )

    # 3. Trim leading and trailing silence
    if trim:
        audio, _ = librosa.effects.trim(
            audio,
            top_db=TOP_DB,
        )

    if len(audio) == 0:
        raise ValueError(f"Empty audio after preprocessing: {file_path}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio values: {file_path}")

    # 4. Optional peak normalization
    if normalize:    
        peak = np.max(np.abs(audio))

        if peak > 1e-8:
            audio = audio / peak

    return audio, TARGET_SR


if __name__=="__main__":

    # Taking a single sample file first to extract the features
    sample_file = (
        PROJECT_ROOT
        / "data/raw/Actor_01/03-01-01-01-01-01-01.wav"
    )

    # Load the original waveform from below we will get -- 
    # audio : amplitude of audio signal at each sampled instant
    # sr : The sampling rate, measured in samples per second

    audio, sr = librosa.load(
        sample_file,
        sr=None,
        mono=True,
    )

    # As checked with mono=False, the audio save only 1 channel, which signifies our audio data is Mono audio
    # print("Audio shape:", audio.shape)
    # print("Number of dimensions:", audio.ndim)
    # print(sr) # 48000 samples per second in original audio

    # Now we will resample it to 16KHz (As per Nyquist frequency it preserves the frequency information below 8KHz (which is required for speech-emotion recognition))
    # It preserves the useful speech information and reduce computational cost

    resampled_audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=TARGET_SR,
    )
    validate_resampling(audio,sr,resampled_audio,TARGET_SR)

    # Now we will proceed with Silence Trimming (To remove the redundant and un-usable samples)
    # RAVDESS recordings often contain silence or very quiet regions before and after the actor speaks.
    # For emotion recognition we only need to analyze the speech signal

    # Choosing top_db as 35 dB, cause its practical threshold that aims to remove quiet leading and trailing regions without removing too much speech

    trimmed_audio, trim_indices = librosa.effects.trim(
        resampled_audio,
        top_db=TOP_DB,
    )
    validate_trimming(
        resampled_audio,
        trimmed_audio,
        trim_indices,
    )
    # original_duration = len(resampled_audio) / TARGET_SR
    # trimmed_duration = len(trimmed_audio) / TARGET_SR

    # retained_percentage = (
    #     100 * trimmed_duration / original_duration
    # )
    # # We can see After Trimming the duration decreased from 3.3 seconds to 1.5 seconds (duration retained is 46%)
    # print("Before trimming:", original_duration, "seconds")
    # print("After trimming:", trimmed_duration, "seconds")
    # # print("Retained percentage:", retained_percentage)
    # # print("Trim indices:", trim_indices)
