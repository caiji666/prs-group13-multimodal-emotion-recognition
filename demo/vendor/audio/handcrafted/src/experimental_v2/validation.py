import numpy as np
import pandas as pd
import librosa


def validate_resampling(
    original_audio,
    original_sr,
    resampled_audio,
    target_sr,
):
    """Validate audio resampling results."""

    # Expected number of samples
    expected_samples = round(
        len(original_audio) * target_sr / original_sr
    )

    assert abs(len(resampled_audio) - expected_samples) <= 1

    # Check duration preservation
    original_duration = len(original_audio) / original_sr
    resampled_duration = len(resampled_audio) / target_sr

    assert abs(original_duration - resampled_duration) <= 1 / target_sr

    # Check numerical validity
    assert len(resampled_audio) > 0
    assert np.isfinite(resampled_audio).all()

    print("Resampling validation passed!")


def validate_trimming(
    resampled_audio,
    trimmed_audio,
    trim_indices,
):
    """Validate the trimmed waveform."""

    start, end = trim_indices

    assert 0 <= start <= end <= len(resampled_audio)

    assert len(trimmed_audio) == end - start
    assert len(trimmed_audio) > 0

    assert np.isfinite(trimmed_audio).all()

    print("Silence trimming validation passed!")


def validate_preprocessing_dataset(
    metadata,
    project_root,
    preprocess_fn,
):
    """Validate preprocessing across all recordings."""

    records = []

    for row in metadata.itertuples(index=False):

        file_path = project_root / row.file_path

        try:
            # Original waveform
            original_audio, original_sr = librosa.load(
                file_path,
                sr=None,
                mono=True,
            )

            # Without normalization
            audio_raw, sr = preprocess_fn(
                file_path,
                normalize=False,
                trim=True,
            )

            # With normalization
            audio_norm, norm_sr = preprocess_fn(
                file_path,
                normalize=True,
                trim=True,
            )

            # Numerical checks
            assert sr == 16000
            assert norm_sr == 16000
            assert len(audio_raw) > 0
            assert len(audio_raw) == len(audio_norm)
            assert np.isfinite(audio_raw).all()
            assert np.isfinite(audio_norm).all()

            original_duration = len(original_audio) / original_sr
            processed_duration = len(audio_raw) / sr

            retained_pct = (
                100 * processed_duration / original_duration
            )

            raw_peak = np.max(np.abs(audio_raw))
            norm_peak = np.max(np.abs(audio_norm))

            if raw_peak > 1e-8:
                np.testing.assert_allclose(
                    norm_peak, 1.0, atol=1e-6
                )

            records.append({
                "sample_id": row.sample_id,
                "actor_id": row.actor_id,
                "emotion_label": row.emotion_label,
                "original_duration": original_duration,
                "processed_duration": processed_duration,
                "retained_pct": retained_pct,
                "raw_peak": raw_peak,
                "normalized_peak": norm_peak,
                "raw_rms": np.sqrt(np.mean(audio_raw ** 2)),
                "normalized_rms": np.sqrt(np.mean(audio_norm ** 2)),
                "status": "success",
                "error": "",
            })

        except Exception as exc:
            records.append({
                "sample_id": row.sample_id,
                "actor_id": row.actor_id,
                "emotion_label": row.emotion_label,
                "status": "failed",
                "error": str(exc),
            })

    return pd.DataFrame(records)
