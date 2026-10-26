
import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_ROOT = (
    PROJECT_ROOT / "outputs/features/audio_fusion"
)

METRICS_ROOT = (
    PROJECT_ROOT / "outputs/metrics/audio_fusion"
)

# The handcrafted dataset does not depend on the M2 source and is
# read by the M5 handover export, so it always stays at this path.
HANDCRAFTED_PATH = FEATURE_ROOT / "handcrafted_48d.csv"


# M2 learned-feature exports used in the M1 representation comparison
M2_SOURCES = {
    # M2 final model: frozen WavLM Base+ layer 6 + temporal adapter
    # + attentive statistics pooling (Round 05 export).
    # Extract m5_learned_feature_only.zip into data/external/.
    "wavlm": {
        "input_dir": (
            PROJECT_ROOT
            / "data/external/m5_learned_feature_only/pooled"
        ),
        "feature_prefix": "F_audio_learned_feature_only_",
        "output_subdir": "",
    },

    # Earlier M2 export: Stage 3 MFCC Transformer (epoch 14).
    # Kept for reproducing the archived comparison.
    "mfcc_transformer": {
        "input_dir": (
            PROJECT_ROOT / "data/external/m2_audio/pooled"
        ),
        "feature_prefix": "f_",
        "output_subdir": "archive/mfcc_transformer",
    },
}

DEFAULT_SOURCE = "wavlm"


def parse_source(description: str) -> str:
    """Read the --m2-source command-line option."""

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--m2-source",
        choices=sorted(M2_SOURCES),
        default=DEFAULT_SOURCE,
        help="M2 learned-feature export to compare against",
    )

    return parser.parse_args().m2_source


def feature_dir(source: str) -> Path:
    """Directory holding the learned and combined datasets."""

    return FEATURE_ROOT / M2_SOURCES[source]["output_subdir"]


def metrics_dir(source: str) -> Path:
    """Directory holding the comparison metrics."""

    return METRICS_ROOT / M2_SOURCES[source]["output_subdir"]
