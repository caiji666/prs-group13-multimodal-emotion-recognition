
import json

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from src.fusion.m2_sources import (
    PROJECT_ROOT,
    HANDCRAFTED_PATH,
    M2_SOURCES,
)
from src.handover.export_m1_audio import (
    EMOTIONS,
    RANDOM_STATE,
    create_model,
)


# M2 WavLM Round 05 export (m5_learned_feature_only)
WAVLM_DIR = M2_SOURCES["wavlm"]["input_dir"].parent

OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/handover/final_audio"
)

# Class order required by multimodal_fusion (fusion.CLASSES).
# M1 calls class 7 "surprised"; the fusion contract uses "surprise".
FUSION_CLASSES = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprise",
)

M1_LABEL_MAP = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}

SPLITS = ["train", "val", "test"]


def load_wavlm():
    """Load M2's per-sample WavLM sequences, pooled features and probabilities."""

    manifest = pd.read_csv(WAVLM_DIR / "manifest.csv")

    assert len(manifest) == 1440
    assert manifest["sample_id"].is_unique

    records = []

    for row in manifest.itertuples(index=False):

        with np.load(
            WAVLM_DIR / row.feature_file,
            allow_pickle=False,
        ) as sample:

            assert str(sample["sample_id"]) == row.sample_id
            assert tuple(sample["class_names"].tolist()) == FUSION_CLASSES

            sequence = sample["F_sequence"]
            length = int(sample["length"])

            assert sequence.shape == (length, 128)
            assert not sample["padding_mask"].any()

            records.append({
                "sample_id": row.sample_id,
                "actor_id": int(sample["actor_id"]),
                "split": str(sample["split"]),
                "label": int(sample["label"]),
                "length": length,
                "sequence": sequence,
                "pooled": sample["F_audio_learned_feature_only"],
                "probs": sample["P_audio_learned_feature_only"],
            })

    # Deterministic order; the fusion loader pairs modalities by ID
    records.sort(key=lambda record: record["sample_id"])

    return records


def load_handcrafted(sample_ids, actors, splits, labels):
    """Load M1 48D features aligned to the WavLM sample order."""

    dataset = pd.read_csv(HANDCRAFTED_PATH)

    feature_columns = [
        col for col in dataset.columns
        if col.startswith("m1_")
    ]

    assert len(dataset) == 1440
    assert len(feature_columns) == 48

    dataset = dataset.set_index("trial_id").loc[sample_ids]

    # Verify actor, split and label agreement with M2
    assert (dataset["actor_id"].to_numpy() == actors).all()
    assert (dataset["split"].to_numpy() == splits).all()

    assert (
        dataset["emotion_label"].map(M1_LABEL_MAP).to_numpy()
        == labels
    ).all()

    features = dataset[feature_columns].to_numpy(np.float64)

    assert np.isfinite(features).all()

    return features, feature_columns


def combined_probabilities(features, labels, actors, splits):
    """LR probabilities: 4-fold actor-grouped OOF for train, full-train model for val/test."""

    train = splits == "train"

    X_train = features[train]
    y_train = labels[train]
    groups = actors[train]

    probabilities = np.zeros(
        (len(labels), len(EMOTIONS)),
        dtype=np.float64,
    )

    # 1. Actor-independent OOF predictions for training recordings
    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = np.zeros((len(y_train), len(EMOTIONS)))
    oof_counts = np.zeros(len(y_train), dtype=np.int32)

    for train_idx, held_idx in cv.split(X_train, y_train, groups):

        assert set(groups[train_idx]).isdisjoint(groups[held_idx])

        model = create_model()
        model.fit(X_train[train_idx], y_train[train_idx])

        # Align probabilities to canonical class order
        oof[np.ix_(held_idx, model.classes_)] = model.predict_proba(
            X_train[held_idx]
        )

        oof_counts[held_idx] += 1

    assert np.all(oof_counts == 1)

    probabilities[train] = oof

    # 2. Final model fitted on all training recordings
    final_model = create_model()
    final_model.fit(X_train, y_train)

    assert np.array_equal(
        final_model.classes_,
        np.arange(len(EMOTIONS)),
    )

    probabilities[~train] = final_model.predict_proba(
        features[~train]
    )

    return probabilities


def validation_scores(probs, labels, splits):
    """Validation-only accuracy and Macro-F1; test is not scored."""

    val = splits == "val"
    predictions = probs[val].argmax(axis=1)

    return {
        "val_accuracy": float(
            accuracy_score(labels[val], predictions)
        ),
        "val_macro_f1": float(
            f1_score(labels[val], predictions, average="macro")
        ),
    }


def save_bundle(path, sample_ids, labels, actors, pooled, probs, sequence, lengths):
    """Write one modality file in the multimodal_fusion NPZ contract."""

    assert pooled.shape[0] == len(sample_ids)
    assert np.isfinite(pooled).all()

    assert probs.shape == (len(sample_ids), 8)
    assert np.isfinite(probs).all()
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    np.savez_compressed(
        path,
        sample_ids=sample_ids,
        labels=labels.astype(np.int64),
        actors=actors.astype(np.int64),
        class_names=np.asarray(FUSION_CLASSES),
        pooled=pooled.astype(np.float32),
        probs=probs.astype(np.float32),
        sequence=sequence,
        lengths=lengths.astype(np.int64),
    )


def main():

    # 1. Load M2 WavLM export
    records = load_wavlm()

    sample_ids = np.asarray([r["sample_id"] for r in records])
    actors = np.asarray([r["actor_id"] for r in records])
    splits = np.asarray([r["split"] for r in records])
    labels = np.asarray([r["label"] for r in records])
    lengths = np.asarray([r["length"] for r in records])

    # Fusion requires labels and actors consistent with the trial ID
    assert np.array_equal(
        labels,
        [int(sid.split("-")[1]) - 1 for sid in sample_ids],
    )
    assert np.array_equal(
        actors,
        [int(sid.split("-")[-1]) for sid in sample_ids],
    )

    split_counts = {
        split: int((splits == split).sum())
        for split in SPLITS
    }

    assert split_counts == {"train": 960, "val": 240, "test": 240}

    # 2. Sequence: M2's F_sequence unchanged, zero-padded at the end
    max_length = int(lengths.max())

    sequence = np.zeros(
        (len(records), max_length, 128),
        dtype=np.float32,
    )

    for index, record in enumerate(records):
        sequence[index, :record["length"]] = record["sequence"]

    wavlm_pooled = np.stack([r["pooled"] for r in records])
    wavlm_probs = np.stack([r["probs"] for r in records])

    # 3. M1 handcrafted features aligned by trial ID
    handcrafted, handcrafted_columns = load_handcrafted(
        sample_ids,
        actors,
        splits,
        labels,
    )

    # 4. Variant A: M2 WavLM delivery repackaged (selected representation)
    # 5. Variant B: handcrafted + WavLM pooled + M1 LR probabilities
    combined_pooled = np.hstack([handcrafted, wavlm_pooled])

    combined_probs = combined_probabilities(
        combined_pooled,
        labels,
        actors,
        splits,
    )

    variants = {
        "audio_wavlm.npz": {
            "role": "optional single-file copy of M2's WavLM delivery (selected representation)",
            "pooled": wavlm_pooled,
            "probs": wavlm_probs,
            "pooled_columns": [f"m2_f_{i:03d}" for i in range(128)],
            "pooled_source": "M2 WavLM Round 05 attentive-statistics pooled representation",
            "probs_source": "M2 WavLM Round 05 MLP head (P_audio_learned_feature_only)",
            "train_probability_method": "In-sample (M2 model trained on these actors)",
        },
        "audio_combined.npz": {
            "role": "optional comparison variant: handcrafted + WavLM concatenation",
            "pooled": combined_pooled,
            "probs": combined_probs,
            "pooled_columns": [
                *handcrafted_columns,
                *[f"m2_f_{i:03d}" for i in range(128)],
            ],
            "pooled_source": "M1 48D handcrafted (db2, norm, 20 ms) || M2 WavLM 128D pooled",
            "probs_source": "M1 StandardScaler + LogisticRegression (C=1.0) on the 176D vector",
            "train_probability_method": (
                "4-fold actor-independent OOF; WavLM features of training "
                "actors were seen by M2's adapter, so these remain optimistic"
            ),
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_variants = {}

    for filename, variant in variants.items():

        save_bundle(
            OUTPUT_DIR / filename,
            sample_ids,
            labels,
            actors,
            variant["pooled"],
            variant["probs"],
            sequence,
            lengths,
        )

        scores = validation_scores(
            variant["probs"],
            labels,
            splits,
        )

        metadata_variants[filename] = {
            "role": variant["role"],
            "pooled_dimensions": int(variant["pooled"].shape[1]),
            "pooled_columns": variant["pooled_columns"],
            "pooled_source": variant["pooled_source"],
            "probs_source": variant["probs_source"],
            "train_probability_method": variant["train_probability_method"],
            "val_test_probability_method": "Full-training fitted model",
            **scores,
        }

        print(
            f"{filename}: pooled {variant['pooled'].shape[1]}D | "
            f"Val Accuracy: {scores['val_accuracy']:.4f} | "
            f"Val Macro-F1: {scores['val_macro_f1']:.4f}"
        )

    # 6. Metadata
    wavlm_metadata = json.loads(
        (WAVLM_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    metadata = {
        "format": "multimodal_fusion NPZ contract (fusion.data.load_modality)",
        "fields": {
            "sample_ids": "[N] six-field trial IDs",
            "labels": "[N] int64, RAVDESS emotion code - 1",
            "actors": "[N] int64",
            "class_names": "[8]",
            "pooled": "[N, D] float32",
            "probs": "[N, 8] float32 probabilities",
            "sequence": f"[N, {max_length}, 128] float32, M2 F_sequence zero-padded at the end",
            "lengths": "[N] int64 valid sequence steps",
        },
        "class_names": list(FUSION_CLASSES),
        "selection": (
            "WavLM learned-only achieved the highest validation Macro-F1 in the "
            "M1 representation comparison (docs/04), so the audio input for "
            "fusion is M2's WavLM delivery (m5_learned_feature_only), loaded "
            "directly by M5 with fusion/audio_pilot.py. Both files here are "
            "optional: audio_wavlm.npz repackages M2's delivery in the "
            "single-file fusion format, and audio_combined.npz allows a "
            "fusion-level confirmation of the selection."
        ),
        "variants": metadata_variants,
        "sequence_source": "M2 WavLM Round 05 temporal adapter sequence (F_sequence), unchanged",
        "sequence_length": {
            "min": int(lengths.min()),
            "max": max_length,
            "mean": float(lengths.mean()),
        },
        "m2_checkpoint_sha256": wavlm_metadata["checkpoint_sha256"],
        "split_counts": split_counts,
        "test_scored": False,
        "notes": [
            "Train fusion models on train, tune weights on val and keep test for final evaluation.",
            "Training-set audio probabilities are over-confident in both variants; tune decision-fusion weights on validation.",
            "M1 class 7 is renamed from 'surprised' to the fusion spelling 'surprise'.",
        ],
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nFinal audio export completed!")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
