# Visual Deep Learning for Emotion Recognition

Distribution: generated features stay local; trained weights are restored from the separate model ZIP. See [feature generation](../../docs/feature-generation.md) and the root README for setup.

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

## Run the notebooks

Use a separate Python environment with the TensorFlow dependencies in this folder:

```bash
python -m venv .venv-m4
source .venv-m4/bin/activate
python -m pip install -r visual/visual_deep_learning/requirements.txt
python -m jupyterlab visual/visual_deep_learning
```

The supplied models were saved with Keras 3.15.1. Face sequences must be prepared with M3 before training or raw-frame extraction. Saved feature exports can be inspected directly without retraining. The integrated video interface uses M3 handcrafted visual extraction and PyTorch fusion; the original M4 TensorFlow models are operated through these notebooks.
