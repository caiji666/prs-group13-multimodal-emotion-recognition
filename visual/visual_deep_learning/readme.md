# Visual Deep Learning for Emotion Recognition

This folder contains the deep learning visual models (M4) developed for emotion recognition using the RAVDESS dataset.

## Overview

Each video uses 16 preprocessed RGB face frames (224 × 224), provided by M3. Three visual models were implemented:

- **MobileNetV3-Small + Mean Pooling**
- **ResNet18 + Mean Pooling**
- **ResNet18 + BiLSTM**

All models use ImageNet-pretrained CNN backbones with frozen weights and predict eight emotion classes. MobileNetV3-Small with mean pooling was selected based on validation Macro-F1.

## File Structure

- `faces/` — Preprocessed 16-frame face sequences from M3.
- `actor_splits.csv` — Actor-independent train, validation and test splits.
- `*.ipynb` — Model implementation, training and feature extraction.
- `*.csv` — Training logs for each model.
- `*.keras` — Saved model checkpoints.
- `*_features.npz` — Exported visual features and prediction probabilities.

## Validation Results

| Model | Best Validation Macro-F1 |
|---|---:|
| MobileNetV3-Small + Mean Pooling | **0.3771** |
| ResNet18 + Mean Pooling | 0.2099 |
| ResNet18 + BiLSTM | 0.2128 |

## Feature Export

Each `*_features.npz` file contains sample IDs, labels, dataset splits, frame-level features, pooled video-level features, prediction probabilities and model metadata.

These outputs can be used for subsequent multimodal fusion.