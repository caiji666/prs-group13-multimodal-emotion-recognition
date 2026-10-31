# Evaluation protocol

## Dataset and partitions

RAVDESS speech contains 1,440 paired recordings from 24 actors and eight classes. The shared partition is defined in [`actor_splits.csv`](../multimodal_fusion/configs/actor_splits.csv).

| Split | Actors | Recordings |
|---|---|---:|
| Train | 01, 04, 05, 06, 07, 10, 11, 12, 13, 15, 16, 18, 19, 20, 23, 24 | 960 |
| Validation | 09, 14, 21, 22 | 240 |
| Test | 02, 03, 08, 17 | 240 |

Classes are ordered `neutral, calm, happy, sad, angry, fearful, disgust, surprise` (labels 0–7). Input class name `surprised` is normalized to `surprise`.

A paired trial ID removes only the modality prefix and file extension: `03-01-06-01-02-01-12.wav` becomes `01-06-01-02-01-12`. Leading zeros are retained. Modalities are joined by ID, with labels, actors, uniqueness and partitions checked before training.

## Final representations

- **Audio:** selected M2 WavLM refinement R05, run `M2_encoder_wavlm_refinement_s42_20261021_05`; pooled and sequence dimensions are 128. Pooling uses the delivered attentive statistics head.
- **Visual:** selected handcrafted Haar DWT representation, 320 dimensions over 16 frames. Deep 576D and combined 896D representations form separate ablations.
- **Audio alternatives:** handcrafted 48D and combined 176D controls are identified separately.
- **Feature fitting:** normalizer statistics come from the training partition. The upstream encoders stay frozen during fusion training.

M1's WavLM representation comparison uses its exported W05 feature set; the final M5 study uses the later R05 delivery. Their recorded validation values refer to those respective inputs.

## Selection and reporting

Neural models use validation Macro-F1 for early stopping and checkpoint selection. The final study uses seeds 42, 43 and 44, AdamW, learning rate 0.001, batch size 64, up to 30 epochs and patience 8. Decision fusion searches audio weights from 0 to 1 in steps of 0.05 using validation Macro-F1.

The predeclared final study contains 93 neural fits, 27 SVM configurations and 10 fixed pipelines. Among concatenation, gated fusion and cross-attention on the primary representations, concatenation has the highest mean validation Macro-F1. `evaluation_lock.json` preserves the selected method, input hashes and checkpoint hashes before final test inference.

Test output reports accuracy, Macro-F1, macro precision/recall, class metrics and confusion matrix counts. SVM configurations with `probability=False` supply decision scores separately. The original delivered visual probability argmax yields test Macro-F1 0.4570; M3's direct classifier predictions report 0.4718. Those outputs retain separate identities.

The same test actors had been observed during earlier development. Final rerun scores describe this fixed split; three-seed variability measures training variation within the split. Audio training probabilities are in-sample, so train-time probability stacking is not used. Sequence attention operates on relative token positions, without verified physical audio/video timestamp alignment.

## Video interface and external pilot

The interface uses fixed seed42 models and matching feature dimensions. Raw-video decision fusion uses an M5 visual MLP; validation replay can use the delivered visual SVM probabilities. These have separate protocol names and weights.

CREMA-D evaluation uses 24 fixed clips, four actors, one sentence and six reference emotions. Models remain fixed, eight output classes remain available, and no external-score fitting occurs. External accuracy and six-reference-class Macro-F1 are reported separately from RAVDESS. See [the external report](../demo/external_evaluation_20261027/REPORT.md).

Probabilities describe model output and are uncalibrated. Gate values, attention maps and input replacements expose model behavior; their definitions are in [demo methods](demo-methods.md).
