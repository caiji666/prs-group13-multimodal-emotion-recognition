# Generate features locally

Feature arrays, feature tables, extracted faces and caches are generated on the receiving machine. The repository retains their extraction code and schemas. Restore model weights before using learned extractors.

## A video clip

```bash
python project.py extract --video /path/to/video.mp4 --output data/my_clip
```

This produces WavLM audio pooled/sequence features, 16×320 Haar DWT visual features and extraction metadata. It uses the same extractor as the video interface.

## M1 handcrafted audio handoff

For the original 48D handcrafted representation used by the M5 controls:

```bash
python scripts/export_handcrafted_audio.py \
  --audio-dir /path/to/RAVDESS/Audio_Speech_Actors_01-24 \
  --output features/Audio_Handoff/features/handcrafted_features.csv
```

The exporter preserves the recorded 48-column order and shared actor split. It requires the full 1,440-sample speech subset by default. `--smoke` allows a small subset for extraction checks.

M1's separate configurable preprocessing studies are retained under `audio/handcrafted/src/experimental_v2/`: `extract_time_dataset`, `extract_spectral_dataset`, `extract_wavelet_dataset` and `combine_features`. The [M1 guide](../audio/handcrafted/README.md) describes those configurations and classifier comparisons.

## M2 selected WavLM R05 export

The original refinement exporter generates per-sample sequences, pooled features, probabilities, metadata and checksums:

```bash
cd audio/learned
python -m research.speech_transfer.download
python -m research.speech_transfer.refinement.export_delivery \
  --checkpoint ../../experiment_records/M2_encoder_wavlm_refinement_s42_20261021_05/best.pt \
  --audio-root /path/to/RAVDESS/Audio_Speech_Actors_01-24 \
  --output ../../features/r05
cd ../..
```

The output consumed by fusion is `features/r05/m5_learned_feature_only/`. The exporter preserves all 1,440 sample IDs and shared partitions. Download original speech audio using `python audio/learned/scripts/download_data.py` if needed.

## M3 handcrafted visual features

In the M3 environment, from `visual/handcrafted/`:

```bash
python -m m3.extract_faces --video-dir /path/to/RAVDESS --out-dir data/faces --modality 01
python -m m3.dwt_features --faces-dir data/faces --out data/dwt_features.npz
python -m m3.train_svm --features data/dwt_features.npz --out-dir outputs
```

This generates the aligned face sequences, per-frame DWT features, pooled handcrafted representation and SVM outputs. The [M3 guide](../visual/handcrafted/README.md) supplies augmentation and visual representation comparison commands.

## M4 learned visual features

Use the TensorFlow environment and the three notebooks under `visual/visual_deep_learning/`. Prepare M3's 16-frame face inputs, restore the matching `.keras` weights, and run the extraction/export cells to generate each model's `*_features.npz`. See the [M4 guide](../visual/visual_deep_learning/readme.md).

M3's `m3.standardize_m4_visual_outputs` and `m3.run_visual_representation_comparison` create the combined handoff with HC/deep/combined representations and fusion-safe probabilities. Their CLI help exposes the source paths, source model, augmentation and split checks. This is the full visual comparison/export workflow, separate from the raw-video UI's handcrafted extraction path.

## Fusion development cache

Once the full audio, visual comparison and handcrafted audio exports exist:

```bash
python scripts/prepare_development.py \
  --audio features/r05/m5_learned_feature_only \
  --visual visual/handcrafted/outputs_visual_comparison/visual_representation_outputs.npz \
  --handcrafted features/Audio_Handoff/features/handcrafted_features.csv
```

This validates IDs, classes, actors, split counts and source checksums, then writes only train/validation samples to the local development cache. The cache enables replay and the final fusion training command. Test samples remain in the original generated handoffs for the study's evaluation stage.
