# WavLM continuation 20261021 Round05 (residual temporal convolution kernel3) audio handoff for M5

The directory/NPZ field contract follows the previous M2 learned-only delivery. Each per_sample/{train,val,test}/{sample_id}.npz contains F_sequence [T,128], F_audio_learned_feature_only [128], P_audio_learned_feature_only [8], sample_id, actor_id, split, label, length, padding_mask and class_names. Load with numpy.load(path, allow_pickle=False). manifest.csv indexes every sample. pooled/{split}.npz and CSV contain aligned pooled/probability tables.

F_sequence is M2's trained temporal adapter sequence for cross-attention. F_audio_learned_feature_only is the 128-dimensional attentive-statistics pooled representation, without M1 handcrafted features. It is NOT sequence.mean(0). P_audio_learned_feature_only is softmax of this M2 model's MLP; it is NOT M1's final audio classifier probability.

Match six-field sample_id, label, split and class order before fusion. Class order: neutral, calm, happy, sad, angry, fearful, disgust, surprise (labels 0..7). Pad variable sequences when batching; True indicates padding, and stored individual masks are all False. WavLM sequence time steps differ from MFCC, and boundary silence trimming must be considered for precise audio/video alignment.

The backbone and selected Round 05 adapter were fixed before export; no augmentation, training, model selection or test metrics computation occurred during export. Test WAVs were decoded solely to generate the held-out fusion inputs. Train fusion on train, select with val and reserve test for final evaluation. Train probabilities are in-sample, not out-of-fold. See metadata.json for provenance; checksums.json lists per-sample SHA-256.

## Current model and evaluation

This package replaces the earlier WavLM delivery with continuation20261021 Round05, including kernel3 residual temporal convolution. Validation Accuracy84.1667%, Macro-F10.831218. Separate test report is in the parent folder final_test_evaluation. Feature files preserve the old contract. No M1 handcrafted features are included.
