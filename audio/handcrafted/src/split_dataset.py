from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.csv"

# Load data/metadata.csv
metadata = pd.read_csv(METADATA_PATH)

# print(metadata.shape)
# print(metadata.head())

# Identify the unique actors
actors = np.sort(metadata["actor_id"].unique())

# print("Actors:", actors)
# print("Total actors:", len(actors))
assert np.array_equal(actors, np.arange(1, 25))

male_actors = actors[actors % 2 == 1]
female_actors = actors[actors % 2 == 0]

rng = np.random.default_rng(42)

rng.shuffle(male_actors)
rng.shuffle(female_actors)


# Divide the shuffled actors - Keeping Gender Division Equal
# Training - 8 Male and 8 Female
train_actors = np.concatenate([
    male_actors[:8],
    female_actors[:8],
])
# Validation - 2 Male and 2 Female
val_actors = np.concatenate([
    male_actors[8:10],
    female_actors[8:10],
])
# Test - 2 Male and 2 Female
test_actors = np.concatenate([
    male_actors[10:],
    female_actors[10:],
])

train_actors = np.sort(train_actors)
val_actors = np.sort(val_actors)
test_actors = np.sort(test_actors)

print("Train:", train_actors)
print("Validation:", val_actors)
print("Test:", test_actors)

# assert len(train_actors) == 16
# assert len(val_actors) == 4
# assert len(test_actors) == 4

# train_set = set(train_actors)
# val_set = set(val_actors)
# test_set = set(test_actors)

# assert train_set.isdisjoint(val_set)
# assert train_set.isdisjoint(test_set)
# assert val_set.isdisjoint(test_set)

# assert train_set | val_set | test_set == set(actors)

# print("All actor split checks passed!")


# Create the actor-to-split dictionary
actor_to_split = {}

for actor in train_actors:
    actor_to_split[int(actor)] = "train"

for actor in val_actors:
    actor_to_split[int(actor)] = "val"

for actor in test_actors:
    actor_to_split[int(actor)] = "test"

print(actor_to_split)


# Generate actor_splits.csv

actor_records = []

for actor in sorted(actor_to_split):

    record = {
        "actor_id": actor,
        "gender": "male" if actor % 2 == 1 else "female",
        "split": actor_to_split[actor],
    }

    actor_records.append(record)

actor_splits = pd.DataFrame(actor_records)

print(actor_splits.head())


# Verify gender balance

# gender_distribution = pd.crosstab(
#     actor_splits["split"],
#     actor_splits["gender"],
# )

# print(gender_distribution)

# Save the actor_splits.csv
actor_splits.to_csv(
    PROJECT_ROOT / "data/actor_splits.csv",
    index=False,
)


# Generate Audio Split
metadata["split"] = metadata["actor_id"].map(actor_to_split)

assert metadata["split"].notna().all()

audio_splits = metadata[
    [
        "sample_id",
        "actor_id",
        "emotion_label",
        "split",
    ]
].copy()

print(audio_splits.head())

audio_splits.to_csv(
    PROJECT_ROOT / "data/audio_splits.csv",
    index=False,
)
