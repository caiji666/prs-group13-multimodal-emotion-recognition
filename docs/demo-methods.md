# Video extraction and explanations

## Feature extraction

Audio is decoded by FFmpeg to float32 WAV, then loaded as mono 16 kHz and trimmed at 35 dB using the upstream M1/M2 preprocessing. The pinned `microsoft/wavlm-base-plus` revision is `4c66d4806a428f2e922ccfa1a962776e232d487b`. Layer6 features follow the training cache's float16 / float32 conversion, then the selected R05 temporal adapter, attentive statistics pooling and classifier. The pooled vector is 128D.

For handcrafted audio controls, the pinned M1 extraction function supplies 48 columns in the delivery's feature order. Combined audio concatenates HC48 followed by WavLM128. The interface exposes only checkpoints trained for the selected representation.

Visual extraction uniformly samples 16 frames, uses MediaPipe model0 / confidence0.5, aligns by the eyes, applies a 1.3 face-box margin and produces 224px RGB crops. Haar DWT on the inner 180px crop uses a 4×4 grid and five statistics per subband, yielding a 16×320 sequence and a 320D temporal mean. When individual boxes are missing, the nearest detected box is reused; no detected faces produces a reported center crop.

The source snapshot is identified by `demo/vendor/manifest.json`. The WavLM backbone SHA256 is `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`; the audio adapter SHA256 is `0db8f833f3d32f21b676c7ed008acdf094bb20dda31e3498e642249d597f04bb`.

The uploaded file's SHA256 identifies the clip. Estimated audio token centers and video frame times are preserved in the result. Video audio encoding can differ from the audio-only dataset files; raw-video feature parity with every original RAVDESS export has not been established.

## Explanation definitions

| Method | Display | Interpretation |
|---|---|---|
| Decision | Per-class weighted audio and visual terms | Exact decomposition of the fused probability |
| Concat / LMF / unimodal controls | Probability change when one standardized input is replaced by zero | Fixed-target sensitivity to replacing that branch with its train mean |
| Gated | 64 gate values and a fixed-0.5 comparison | Relative mixing of projected branches |
| Cross-attention | Four-head mean audio-query / visual-key attention | Internal token association; up to 48 audio bins displayed |
| Bidirectional attention | Forward heatmap plus reverse summary | Associations in both directions |
| Audio residual | Original audio logits and signed correction logits | The added branch reads both modalities and changes class logits |

Attention export is checked to preserve logits. Gate values are mixing coefficients, and train-mean replacements can be out of distribution. These displays describe the model's computation; model probabilities remain uncalibrated.
