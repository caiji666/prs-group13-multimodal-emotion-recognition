# Local feature workspace

Feature files are generated locally and excluded from Git and source/submission ZIPs. See [the feature generation guide](../docs/feature-generation.md) for the audio, visual and fusion-cache commands.

The feature extraction implementations remain in the M1–M4 modules and `demo/extraction.py`. `project.py extract` exports features for a video clip. `scripts/export_handcrafted_audio.py` exports the M1 handoff, the M2 refinement exporter produces WavLM features, and `scripts/prepare_development.py` builds the optional fusion development cache.
