
from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_audio
from src.experimental_v2.validation import validate_preprocessing_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

metadata = pd.read_csv(
    PROJECT_ROOT / "data/metadata.csv"
)

results = validate_preprocessing_dataset(
    metadata=metadata,
    project_root=PROJECT_ROOT,
    preprocess_fn=preprocess_audio,
)

print("Total recordings:", len(results))
print("\nProcessing status:")
print(results["status"].value_counts())

successful = results[
    results["status"] == "success"
]

print("\nDuration summary:")
print(
    successful[
        [
            "original_duration",
            "processed_duration",
            "retained_pct",
        ]
    ].describe()
)

print("\nRecordings retaining less than 30%:")
print(
    successful[
        successful["retained_pct"] < 30
    ][["sample_id", "retained_pct"]]
)

# Save validation report
output_path = (
    PROJECT_ROOT
    / "outputs/preprocessing_validation.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(output_path, index=False)

print("\nSaved:", output_path)

assert len(results) == 1440
assert (results["status"] == "success").all()

print("\nAll preprocessing checks passed!")
