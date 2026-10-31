# Sample ID Convention for RAVDESS Dataset

## 1. Purpose

To ensure consistent data alignment across all project members (M1–M5), we should use a **standardized Sample ID** for every recording in the project's RAVDESS **speech subset**.

This allows us to correctly match handcrafted audio features, deep-learned audio features, visual features, and emotion labels during feature concatenation and multimodal fusion.

Official RAVDESS filenames already contain fields that identify each recording. However, RAVDESS does not separately provide the cross-modal `sample_id` field defined for this project. Our team derives it automatically from the official filename; **no manual naming of individual samples is needed**.

## 2. Sample ID Generation

RAVDESS filenames follow this structure:

`Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor`

Generate the Sample ID by **removing the first field (Modality)** and the file extension, while keeping the remaining six fields in their original order:

`VocalChannel-Emotion-Intensity-Statement-Repetition-Actor`

**Example:**

```text
Original filename:
03-01-06-01-02-01-12.wav

Generated Sample ID:
01-06-01-02-01-12
```

Corresponding audio-only and audio-visual versions of the same recording therefore share the same Sample ID.

**Python implementation:**

```python
from pathlib import Path


def generate_sample_id(filename):
    parts = Path(filename).stem.split("-")
    if len(parts) != 7 or any(
        len(part) != 2 or not part.isdigit() for part in parts
    ):
        raise ValueError("Expected a RAVDESS filename with seven two-digit fields.")
    if parts[1] != "01":
        raise ValueError("This project convention applies to speech samples only.")
    return "-".join(parts[1:])


# generate_sample_id("03-01-06-01-02-01-12.wav")
# Returns: "01-06-01-02-01-12"
```

All members should use the same function to generate Sample IDs automatically. Preserve the leading zeros.

## 3. Dataset Split

We have already agreed on the actor-independent Train / Validation / Test partition using `actor_splits.csv`.

Assign each sample to its dataset split based on its **Actor ID**, which is the last field of the RAVDESS filename. All samples from the same actor must follow the agreed CSV assignment.

No additional random splitting should be performed.

## 4. Required Metadata

When saving extracted features or model outputs, each member should include:

| Field | Description |
|---|---|
| `sample_id` | Shared recording identifier derived using the rule above |
| `split` | `train`, `val`, or `test`, as determined by `actor_splits.csv` |
| `label` | Emotion class using the team's agreed numeric mapping and class order |
| `features` | Extracted feature vector or sequence |

For prediction outputs, also preserve the agreed emotion class order in probability vectors.

## 5. Important Notes

- This convention applies only to the **speech subset used in this project**.
- Use the **same Sample ID generation rule** across all modules and preserve IDs when saving and loading outputs.
- Match features by `sample_id`, rather than assuming that file or row order is identical across members.
- Verify that matched audio and visual representations belong to the same underlying recording.
- Use the same emotion label mapping, class order, and `actor_splits.csv` across all members.
- Require one entry per `sample_id` within each modality's feature table. The same ID is expected to appear across corresponding modalities; it is not a unique identifier for every physical file.
- Before fusion, check for duplicate or missing IDs and confirm that labels and splits agree for every matched sample.

## Expected Outcome

By following this convention, all members can reliably align their extracted features and model outputs, enabling accurate feature concatenation and multimodal fusion.
