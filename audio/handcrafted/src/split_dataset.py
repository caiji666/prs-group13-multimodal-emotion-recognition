
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

METADATA_PATH = DATA_DIR / "metadata.csv"
ACTOR_SPLITS_PATH = DATA_DIR / "actor_splits.csv"
SPLITS_PATH = DATA_DIR / "audio_splits.csv"

RANDOM_SEED = 42


def create_actor_splits(actor_ids: list[int]) -> pd.DataFrame:
    """Create gender-balanced, actor-independent partitions."""

    actors = sorted(int(actor) for actor in actor_ids)

    if len(actors) != 24 or set(actors) != set(range(1, 25)):
        raise ValueError("Expected actor IDs 1 through 24")

    # RAVDESS: odd actor IDs are male, even IDs are female.
    male_actors = np.array([a for a in actors if a % 2 == 1])
    female_actors = np.array([a for a in actors if a % 2 == 0])

    rng = np.random.default_rng(RANDOM_SEED)

    rng.shuffle(male_actors)
    rng.shuffle(female_actors)

    assignments = []

    for group in (male_actors, female_actors):
        for actor in group[:8]:
            assignments.append((int(actor), "train"))

        for actor in group[8:10]:
            assignments.append((int(actor), "val"))

        for actor in group[10:12]:
            assignments.append((int(actor), "test"))

    return pd.DataFrame(
        assignments, columns=["actor_id", "split"]
    ).sort_values("actor_id").reset_index(drop=True)


def validate_splits(df: pd.DataFrame) -> None:
    """Validate split sizes, actor separation and gender balance."""

    expected_counts = {"train": 960, "val": 240, "test": 240}
    expected_actors = {"train": 16, "val": 4, "test": 4}

    counts = df["split"].value_counts().to_dict()
    actor_counts = df.groupby("split")["actor_id"].nunique().to_dict()

    assert counts == expected_counts, (
        f"Unexpected recording counts: {counts}"
    )

    assert actor_counts == expected_actors, (
        f"Unexpected actor counts: {actor_counts}"
    )

    # Each actor must belong to exactly one partition.
    assert df.groupby("actor_id")["split"].nunique().eq(1).all(), (
        "Actor leakage detected"
    )

    # Verify gender balance in each partition.
    unique_actors = df[["actor_id", "split"]].drop_duplicates().copy()
    unique_actors["gender"] = np.where(
        unique_actors["actor_id"] % 2 == 1, "male", "female"
    )

    gender_counts = pd.crosstab(
        unique_actors["split"], unique_actors["gender"]
    )

    assert gender_counts.loc["train"].to_dict() == {
        "female": 8, "male": 8
    }
    assert gender_counts.loc["val"].to_dict() == {
        "female": 2, "male": 2
    }
    assert gender_counts.loc["test"].to_dict() == {
        "female": 2, "male": 2
    }

    print("\nRecording counts:")
    print(df["split"].value_counts().sort_index())

    print("\nUnique actors per split:")
    print(df.groupby("split")["actor_id"].nunique())

    print("\nGender distribution:")
    print(gender_counts)

    print("\nAll split validation checks passed!")


def main() -> None:
    metadata = pd.read_csv(METADATA_PATH)

    actor_ids = metadata["actor_id"].unique().tolist()
    actor_splits = create_actor_splits(actor_ids)

    # Assign each recording based on its actor.
    splits = metadata.merge(
        actor_splits,
        on="actor_id",
        how="left",
        validate="many_to_one",
    )

    if splits["split"].isna().any():
        raise ValueError("Some recordings have no split assignment")

    validate_splits(splits)

    # Modality-independent trial identifier.
    # Remove the first two filename components:
    # modality and vocal channel.
    # This allows matching audio and video speech trials.
    splits["trial_id"] = splits["sample_id"].apply(
        lambda sample: "-".join(str(sample).split("-")[2:])
    )

    actor_splits.to_csv(ACTOR_SPLITS_PATH, index=False)
    splits.to_csv(SPLITS_PATH, index=False)

    print("\nActor assignments:")
    print(actor_splits.to_string(index=False))

    print(f"\nSaved: {ACTOR_SPLITS_PATH}")
    print(f"Saved: {SPLITS_PATH}")


if __name__ == "__main__":
    main()
