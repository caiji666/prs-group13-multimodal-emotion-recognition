# Group 13 Experiment Summary — v2

## Real RAVDESS speech — pipeline configurations

| Run ID / record link | Member / stage | Comparison group | Feature extractor(s) | Representation / dimensions | Fusion | Classifier / head | Training regime | Seed |
|---|---|---|---|---|---|---|---|---|
| [M2_encoder_cnn_s42_20261015_00](M2_encoder_cnn_s42_20261015_00/run.json) | M2 / encoder | M2_audio_encoders_20261015_00 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; random initialization | 42 |
| [M2_encoder_bilstm_s42_20261015_00](M2_encoder_bilstm_s42_20261015_00/run.json) | M2 / encoder | M2_audio_encoders_20261015_00 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; random initialization | 42 |
| [M2_encoder_transformer_s42_20261015_00](M2_encoder_transformer_s42_20261015_00/run.json) | M2 / encoder | M2_audio_encoders_20261015_00 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; random initialization | 42 |
| [M2_encoder_cnn_s42_20261017_01](M2_encoder_cnn_s42_20261017_01/run.json) | M2 / encoder | M2_audio_encoders_20261017_01 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; random initialization | 42 |
| [M2_encoder_bilstm_s42_20261017_01](M2_encoder_bilstm_s42_20261017_01/run.json) | M2 / encoder | M2_audio_encoders_20261017_01 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; random initialization | 42 |
| [M2_encoder_transformer_s42_20261017_01](M2_encoder_transformer_s42_20261017_01/run.json) | M2 / encoder | M2_audio_encoders_20261017_01 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; random initialization | 42 |
| [M2_encoder_cnn_s42_20261017_02](M2_encoder_cnn_s42_20261017_02/run.json) | M2 / encoder | M2_audio_encoders_20261017_02 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261017_02](M2_encoder_bilstm_s42_20261017_02/run.json) | M2 / encoder | M2_audio_encoders_20261017_02 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261017_02](M2_encoder_transformer_s42_20261017_02/run.json) | M2 / encoder | M2_audio_encoders_20261017_02 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261017_03](M2_encoder_cnn_s42_20261017_03/run.json) | M2 / encoder | M2_audio_encoders_20261017_03 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261017_03](M2_encoder_bilstm_s42_20261017_03/run.json) | M2 / encoder | M2_audio_encoders_20261017_03 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261017_03](M2_encoder_transformer_s42_20261017_03/run.json) | M2 / encoder | M2_audio_encoders_20261017_03 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261017_04](M2_encoder_cnn_s42_20261017_04/run.json) | M2 / encoder | M2_audio_encoders_20261017_04 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261017_04](M2_encoder_bilstm_s42_20261017_04/run.json) | M2 / encoder | M2_audio_encoders_20261017_04 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261017_04](M2_encoder_transformer_s42_20261017_04/run.json) | M2 / encoder | M2_audio_encoders_20261017_04 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261017_05](M2_encoder_cnn_s42_20261017_05/run.json) | M2 / encoder | M2_audio_encoders_20261017_05 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261017_05](M2_encoder_bilstm_s42_20261017_05/run.json) | M2 / encoder | M2_audio_encoders_20261017_05 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261017_05](M2_encoder_transformer_s42_20261017_05/run.json) | M2 / encoder | M2_audio_encoders_20261017_05 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261018_01](M2_encoder_cnn_s42_20261018_01/run.json) | M2 / encoder | M2_audio_encoders_20261018_01 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261018_01](M2_encoder_bilstm_s42_20261018_01/run.json) | M2 / encoder | M2_audio_encoders_20261018_01 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261018_01](M2_encoder_transformer_s42_20261018_01/run.json) | M2 / encoder | M2_audio_encoders_20261018_01 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.2 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261018_02](M2_encoder_cnn_s42_20261018_02/run.json) | M2 / encoder | M2_audio_encoders_20261018_02 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.35 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261018_02](M2_encoder_bilstm_s42_20261018_02/run.json) | M2 / encoder | M2_audio_encoders_20261018_02 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.35 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261018_02](M2_encoder_transformer_s42_20261018_02/run.json) | M2 / encoder | M2_audio_encoders_20261018_02 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.35 | joint; warm-start | 42 |
| [M2_encoder_cnn_s42_20261018_03](M2_encoder_cnn_s42_20261018_03/run.json) | M2 / encoder | M2_audio_encoders_20261018_03 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_bilstm_s42_20261018_03](M2_encoder_bilstm_s42_20261018_03/run.json) | M2 / encoder | M2_audio_encoders_20261018_03 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_transformer_s42_20261018_03](M2_encoder_transformer_s42_20261018_03/run.json) | M2 / encoder | M2_audio_encoders_20261018_03 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 42 |
| [M2_encoder_cnn_s7_20261018_04](M2_encoder_cnn_s7_20261018_04/run.json) | M2 / encoder | M2_audio_encoders_20261018_04 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 7 |
| [M2_encoder_bilstm_s7_20261018_04](M2_encoder_bilstm_s7_20261018_04/run.json) | M2 / encoder | M2_audio_encoders_20261018_04 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 7 |
| [M2_encoder_transformer_s7_20261018_04](M2_encoder_transformer_s7_20261018_04/run.json) | M2 / encoder | M2_audio_encoders_20261018_04 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.25 | joint; warm-start | 7 |
| [M2_encoder_cnn_s17_20261018_05](M2_encoder_cnn_s17_20261018_05/run.json) | M2 / encoder | M2_audio_encoders_20261018_05 | MFCC → cnn | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 17 |
| [M2_encoder_bilstm_s17_20261018_05](M2_encoder_bilstm_s17_20261018_05/run.json) | M2 / encoder | M2_audio_encoders_20261018_05 | MFCC → bilstm | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 17 |
| [M2_encoder_transformer_s17_20261018_05](M2_encoder_transformer_s17_20261018_05/run.json) | M2 / encoder | M2_audio_encoders_20261018_05 | MFCC → transformer | learned /128 | none | MLP128→64→8, ReLU, dropout0.3 | joint; warm-start | 17 |
| [M5_audio_only_s42_h64_d0.2_20261019](M5_audio_only_s42_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer | combined / [None, 176] | none | Linear(176,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_visual_only_s42_h64_d0.2_20261019](M5_visual_only_s42_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | MobileNetV3-Small per supplied export | learned / [None, 576] | none | Linear(576,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_fusion_concat_s42_20261019_01](M5_fusion_concat_s42_20261019_01/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 752] | concat | Linear(752,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_fusion_gated_s42_20261019_01](M5_fusion_gated_s42_20261019_01/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | gated | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_fusion_cross_attention_s42_20261019_01](M5_fusion_cross_attention_s42_20261019_01/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | cross_attention | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_audio_sequence_mean_s42_h64_d0.2_20261019](M5_audio_sequence_mean_s42_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer | learned / [None, 128] | none | Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_sequence_mean_concat_s42_h64_d0.2_20261019](M5_sequence_mean_concat_s42_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 704] | sequence_mean_concat | Linear(704,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_audio_only_s43_h64_d0.2_20261019](M5_audio_only_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer | combined / [None, 176] | none | Linear(176,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_visual_only_s43_h64_d0.2_20261019](M5_visual_only_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | MobileNetV3-Small per supplied export | learned / [None, 576] | none | Linear(576,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_concat_s43_h64_d0.2_20261019](M5_concat_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 752] | concat | Linear(752,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_gated_s43_h64_d0.2_20261019](M5_gated_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | gated | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_cross_attention_s43_h64_d0.2_20261019](M5_cross_attention_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | cross_attention | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_audio_sequence_mean_s43_h64_d0.2_20261019](M5_audio_sequence_mean_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer | learned / [None, 128] | none | Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_sequence_mean_concat_s43_h64_d0.2_20261019](M5_sequence_mean_concat_s43_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 704] | sequence_mean_concat | Linear(704,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 43 |
| [M5_audio_only_s44_h64_d0.2_20261019](M5_audio_only_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer | combined / [None, 176] | none | Linear(176,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_visual_only_s44_h64_d0.2_20261019](M5_visual_only_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_pooled_controls_v1 | MobileNetV3-Small per supplied export | learned / [None, 576] | none | Linear(576,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_concat_s44_h64_d0.2_20261019](M5_concat_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 752] | concat | Linear(752,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_gated_s44_h64_d0.2_20261019](M5_gated_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | gated | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_cross_attention_s44_h64_d0.2_20261019](M5_cross_attention_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_fusion_mobilenet_audio176_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | cross_attention | Linear(64,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_audio_sequence_mean_s44_h64_d0.2_20261019](M5_audio_sequence_mean_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer | learned / [None, 128] | none | Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_sequence_mean_concat_s44_h64_d0.2_20261019](M5_sequence_mean_concat_s44_h64_d0.2_20261019/run.json) | M5 / fusion | M5_sequence_controls_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 704] | sequence_mean_concat | Linear(704,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 44 |
| [M5_cross_attention_s42_h32_d0.2_20261019](M5_cross_attention_s42_h32_d0.2_20261019/run.json) | M5 / fusion | M5_regularization_exploration_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 32] | cross_attention | Linear(32,32) → ReLU → Dropout(0.2) → Linear(32,8) | frozen_encoder_then_classifier | 42 |
| [M5_cross_attention_s42_h64_d0.4_20261019](M5_cross_attention_s42_h64_d0.4_20261019/run.json) | M5 / fusion | M5_regularization_exploration_v1 | MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 64] | cross_attention | Linear(64,64) → ReLU → Dropout(0.4) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_concat_s42_h32_d0.2_20261019](M5_concat_s42_h32_d0.2_20261019/run.json) | M5 / fusion | M5_regularization_exploration_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 752] | concat | Linear(752,32) → ReLU → Dropout(0.2) → Linear(32,8) | frozen_encoder_then_classifier | 42 |
| [M5_concat_s42_h64_d0.4_20261019](M5_concat_s42_h64_d0.4_20261019/run.json) | M5 / fusion | M5_regularization_exploration_v1 | M1 time/spectral/DWT descriptors; MFCC Transformer; MobileNetV3-Small per supplied export | combined / [None, 752] | concat | Linear(752,64) → ReLU → Dropout(0.4) → Linear(64,8) | frozen_encoder_then_classifier | 42 |
| [M5_reference_audio_head_sNA_20261019_01](M5_reference_audio_head_sNA_20261019_01/run.json) | M5 / reference | M5_imported_probability_references_v1 | referenced upstream classifiers | probabilities / [None, 8] | none | upstream classifier (architecture not supplied) | cached_upstream_evaluation | N/A |
| [M5_reference_visual_head_sNA_20261019_01](M5_reference_visual_head_sNA_20261019_01/run.json) | M5 / reference | M5_imported_probability_references_v1 | referenced upstream classifiers | probabilities / [None, 8] | none | upstream classifier (architecture not supplied) | cached_upstream_evaluation | N/A |
| [M5_fusion_decision_equal_sNA_20261019_01](M5_fusion_decision_equal_sNA_20261019_01/run.json) | M5 / fusion | M5_decision_fixed_exports_v1 | referenced upstream classifiers | probabilities / [None, 8] | decision_equal | none | probability_fusion | N/A |
| [M5_fusion_decision_tuned_sNA_20261019_01](M5_fusion_decision_tuned_sNA_20261019_01/run.json) | M5 / fusion | M5_decision_fixed_exports_v1 | referenced upstream classifiers | probabilities / [None, 8] | decision_tuned | none | probability_fusion | N/A |
| [M5_audio_learned_pilot_s42_20261016_01](M5_audio_learned_pilot_s42_20261016_01/run.json) | M5 / representation | M5_audio_interface_pilot_v1 | MFCC Transformer | learned / [None, 128] | none | Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,8) | frozen_encoder_then_classifier | 42 |

## Real RAVDESS speech — full-pipeline results

| Run ID | Split | N evaluated | Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Evaluation status |
|---|---|---|---|---|---|---|---|
| M2_encoder_cnn_s42_20261015_00 | val | 240 | 0.3836 | 0.4250 | 0.4609 | 0.4219 | completed |
| M2_encoder_cnn_s42_20261015_00 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261015_00 | val | 240 | 0.3593 | 0.4167 | 0.3797 | 0.3906 | completed |
| M2_encoder_bilstm_s42_20261015_00 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261015_00 | val | 240 | 0.4508 | 0.4708 | 0.5393 | 0.4648 | completed |
| M2_encoder_transformer_s42_20261015_00 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261017_01 | val | 240 | 0.4086 | 0.4375 | 0.5073 | 0.4219 | completed |
| M2_encoder_cnn_s42_20261017_01 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261017_01 | val | 240 | 0.3171 | 0.3875 | 0.3363 | 0.3633 | completed |
| M2_encoder_bilstm_s42_20261017_01 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261017_01 | val | 240 | 0.5204 | 0.5292 | 0.5660 | 0.5234 | completed |
| M2_encoder_transformer_s42_20261017_01 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261017_02 | val | 240 | 0.4282 | 0.4458 | 0.4548 | 0.4414 | completed |
| M2_encoder_cnn_s42_20261017_02 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261017_02 | val | 240 | 0.3548 | 0.3917 | 0.3794 | 0.3828 | completed |
| M2_encoder_bilstm_s42_20261017_02 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261017_02 | val | 240 | 0.5296 | 0.5458 | 0.5841 | 0.5391 | completed |
| M2_encoder_transformer_s42_20261017_02 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261017_03 | val | 240 | 0.4126 | 0.4333 | 0.4353 | 0.4297 | completed |
| M2_encoder_cnn_s42_20261017_03 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261017_03 | val | 240 | 0.3914 | 0.4375 | 0.4287 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261017_03 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261017_03 | val | 240 | 0.5263 | 0.5333 | 0.5668 | 0.5352 | completed |
| M2_encoder_transformer_s42_20261017_03 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261017_04 | val | 240 | 0.4241 | 0.4417 | 0.4436 | 0.4414 | completed |
| M2_encoder_cnn_s42_20261017_04 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261017_04 | val | 240 | 0.3869 | 0.4333 | 0.4027 | 0.4219 | completed |
| M2_encoder_bilstm_s42_20261017_04 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261017_04 | val | 240 | 0.5264 | 0.5375 | 0.5655 | 0.5352 | completed |
| M2_encoder_transformer_s42_20261017_04 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261017_05 | val | 240 | 0.4216 | 0.4417 | 0.4646 | 0.4336 | completed |
| M2_encoder_cnn_s42_20261017_05 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261017_05 | val | 240 | 0.3818 | 0.4292 | 0.4185 | 0.4102 | completed |
| M2_encoder_bilstm_s42_20261017_05 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261017_05 | val | 240 | 0.5324 | 0.5458 | 0.5756 | 0.5430 | completed |
| M2_encoder_transformer_s42_20261017_05 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261018_01 | val | 240 | 0.4210 | 0.4375 | 0.4655 | 0.4297 | completed |
| M2_encoder_cnn_s42_20261018_01 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261018_01 | val | 240 | 0.3862 | 0.4333 | 0.4280 | 0.4141 | completed |
| M2_encoder_bilstm_s42_20261018_01 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261018_01 | val | 240 | 0.5324 | 0.5458 | 0.5756 | 0.5430 | completed |
| M2_encoder_transformer_s42_20261018_01 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261018_02 | val | 240 | 0.4234 | 0.4417 | 0.4601 | 0.4375 | completed |
| M2_encoder_cnn_s42_20261018_02 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261018_02 | val | 240 | 0.3901 | 0.4375 | 0.4258 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261018_02 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261018_02 | val | 240 | 0.5243 | 0.5292 | 0.5604 | 0.5273 | completed |
| M2_encoder_transformer_s42_20261018_02 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s42_20261018_03 | val | 240 | 0.4241 | 0.4417 | 0.4660 | 0.4375 | completed |
| M2_encoder_cnn_s42_20261018_03 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s42_20261018_03 | val | 240 | 0.3897 | 0.4375 | 0.4308 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261018_03 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s42_20261018_03 | val | 240 | 0.5292 | 0.5417 | 0.5708 | 0.5391 | completed |
| M2_encoder_transformer_s42_20261018_03 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s7_20261018_04 | val | 240 | 0.4202 | 0.4417 | 0.4664 | 0.4336 | completed |
| M2_encoder_cnn_s7_20261018_04 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s7_20261018_04 | val | 240 | 0.3867 | 0.4333 | 0.4270 | 0.4141 | completed |
| M2_encoder_bilstm_s7_20261018_04 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s7_20261018_04 | val | 240 | 0.5328 | 0.5458 | 0.5763 | 0.5430 | completed |
| M2_encoder_transformer_s7_20261018_04 | test | — | — | — | — | — | not_run |
| M2_encoder_cnn_s17_20261018_05 | val | 240 | 0.4150 | 0.4333 | 0.4413 | 0.4297 | completed |
| M2_encoder_cnn_s17_20261018_05 | test | — | — | — | — | — | not_run |
| M2_encoder_bilstm_s17_20261018_05 | val | 240 | 0.3878 | 0.4292 | 0.4296 | 0.4141 | completed |
| M2_encoder_bilstm_s17_20261018_05 | test | — | — | — | — | — | not_run |
| M2_encoder_transformer_s17_20261018_05 | val | 240 | 0.5286 | 0.5417 | 0.5737 | 0.5391 | completed |
| M2_encoder_transformer_s17_20261018_05 | test | — | — | — | — | — | not_run |
| M5_audio_only_s42_h64_d0.2_20261019 | val | 240 | 0.4645 | 0.4875 | 0.5632 | 0.4727 | completed |
| M5_audio_only_s42_h64_d0.2_20261019 | test | 240 | 0.5643 | 0.5958 | 0.5865 | 0.5781 | completed |
| M5_visual_only_s42_h64_d0.2_20261019 | val | 240 | 0.5114 | 0.5250 | 0.5624 | 0.5391 | completed |
| M5_visual_only_s42_h64_d0.2_20261019 | test | 240 | 0.4012 | 0.3917 | 0.5843 | 0.3945 | completed |
| M5_fusion_concat_s42_20261019_01 | val | 240 | 0.6485 | 0.6542 | 0.6737 | 0.6523 | completed |
| M5_fusion_concat_s42_20261019_01 | test | 240 | 0.6541 | 0.6333 | 0.7414 | 0.6289 | completed |
| M5_fusion_gated_s42_20261019_01 | val | 240 | 0.6043 | 0.6208 | 0.6328 | 0.6055 | completed |
| M5_fusion_gated_s42_20261019_01 | test | 240 | 0.6607 | 0.6792 | 0.7161 | 0.6602 | completed |
| M5_fusion_cross_attention_s42_20261019_01 | val | 240 | 0.6674 | 0.6625 | 0.7100 | 0.6562 | completed |
| M5_fusion_cross_attention_s42_20261019_01 | test | 240 | 0.5671 | 0.5792 | 0.6978 | 0.5586 | completed |
| M5_audio_sequence_mean_s42_h64_d0.2_20261019 | val | 240 | 0.4210 | 0.4458 | 0.4857 | 0.4336 | completed |
| M5_audio_sequence_mean_s42_h64_d0.2_20261019 | test | 240 | 0.5473 | 0.5792 | 0.5760 | 0.5625 | completed |
| M5_sequence_mean_concat_s42_h64_d0.2_20261019 | val | 240 | 0.6245 | 0.6333 | 0.6436 | 0.6250 | completed |
| M5_sequence_mean_concat_s42_h64_d0.2_20261019 | test | 240 | 0.5878 | 0.5750 | 0.6671 | 0.5664 | completed |
| M5_audio_only_s43_h64_d0.2_20261019 | val | 240 | 0.4542 | 0.4792 | 0.4976 | 0.4648 | completed |
| M5_audio_only_s43_h64_d0.2_20261019 | test | 240 | 0.5762 | 0.6000 | 0.6000 | 0.5820 | completed |
| M5_visual_only_s43_h64_d0.2_20261019 | val | 240 | 0.5373 | 0.5500 | 0.5892 | 0.5781 | completed |
| M5_visual_only_s43_h64_d0.2_20261019 | test | 240 | 0.3718 | 0.3875 | 0.5570 | 0.3828 | completed |
| M5_concat_s43_h64_d0.2_20261019 | val | 240 | 0.6477 | 0.6542 | 0.6718 | 0.6523 | completed |
| M5_concat_s43_h64_d0.2_20261019 | test | 240 | 0.6335 | 0.6042 | 0.7386 | 0.5977 | completed |
| M5_gated_s43_h64_d0.2_20261019 | val | 240 | 0.5889 | 0.6083 | 0.6203 | 0.5977 | completed |
| M5_gated_s43_h64_d0.2_20261019 | test | 240 | 0.7090 | 0.7250 | 0.7394 | 0.7070 | completed |
| M5_cross_attention_s43_h64_d0.2_20261019 | val | 240 | 0.6423 | 0.6375 | 0.7049 | 0.6289 | completed |
| M5_cross_attention_s43_h64_d0.2_20261019 | test | 240 | 0.5814 | 0.5667 | 0.7618 | 0.5508 | completed |
| M5_audio_sequence_mean_s43_h64_d0.2_20261019 | val | 240 | 0.4476 | 0.4667 | 0.5237 | 0.4531 | completed |
| M5_audio_sequence_mean_s43_h64_d0.2_20261019 | test | 240 | 0.5130 | 0.5542 | 0.5342 | 0.5352 | completed |
| M5_sequence_mean_concat_s43_h64_d0.2_20261019 | val | 240 | 0.6154 | 0.6208 | 0.6458 | 0.6172 | completed |
| M5_sequence_mean_concat_s43_h64_d0.2_20261019 | test | 240 | 0.6119 | 0.5958 | 0.7161 | 0.5859 | completed |
| M5_audio_only_s44_h64_d0.2_20261019 | val | 240 | 0.4860 | 0.5083 | 0.5613 | 0.4961 | completed |
| M5_audio_only_s44_h64_d0.2_20261019 | test | 240 | 0.5783 | 0.6000 | 0.6053 | 0.5938 | completed |
| M5_visual_only_s44_h64_d0.2_20261019 | val | 240 | 0.5048 | 0.5375 | 0.5524 | 0.5391 | completed |
| M5_visual_only_s44_h64_d0.2_20261019 | test | 240 | 0.4024 | 0.4000 | 0.5858 | 0.3945 | completed |
| M5_concat_s44_h64_d0.2_20261019 | val | 240 | 0.6214 | 0.6333 | 0.6516 | 0.6211 | completed |
| M5_concat_s44_h64_d0.2_20261019 | test | 240 | 0.6563 | 0.6375 | 0.7573 | 0.6289 | completed |
| M5_gated_s44_h64_d0.2_20261019 | val | 240 | 0.6100 | 0.6292 | 0.6591 | 0.6133 | completed |
| M5_gated_s44_h64_d0.2_20261019 | test | 240 | 0.6797 | 0.6792 | 0.7297 | 0.6719 | completed |
| M5_cross_attention_s44_h64_d0.2_20261019 | val | 240 | 0.6728 | 0.6833 | 0.6869 | 0.6797 | completed |
| M5_cross_attention_s44_h64_d0.2_20261019 | test | 240 | 0.5761 | 0.5708 | 0.6852 | 0.5625 | completed |
| M5_audio_sequence_mean_s44_h64_d0.2_20261019 | val | 240 | 0.4289 | 0.4500 | 0.4997 | 0.4375 | completed |
| M5_audio_sequence_mean_s44_h64_d0.2_20261019 | test | 240 | 0.5435 | 0.5708 | 0.5647 | 0.5547 | completed |
| M5_sequence_mean_concat_s44_h64_d0.2_20261019 | val | 240 | 0.6272 | 0.6333 | 0.6631 | 0.6211 | completed |
| M5_sequence_mean_concat_s44_h64_d0.2_20261019 | test | 240 | 0.6174 | 0.6042 | 0.7225 | 0.5938 | completed |
| M5_cross_attention_s42_h32_d0.2_20261019 | val | 240 | 0.6630 | 0.6708 | 0.7430 | 0.6484 | completed |
| M5_cross_attention_s42_h32_d0.2_20261019 | test | — | — | — | — | — | not_run |
| M5_cross_attention_s42_h64_d0.4_20261019 | val | 240 | 0.6828 | 0.6750 | 0.7237 | 0.6680 | completed |
| M5_cross_attention_s42_h64_d0.4_20261019 | test | — | — | — | — | — | not_run |
| M5_concat_s42_h32_d0.2_20261019 | val | 240 | 0.6331 | 0.6500 | 0.6538 | 0.6367 | completed |
| M5_concat_s42_h32_d0.2_20261019 | test | — | — | — | — | — | not_run |
| M5_concat_s42_h64_d0.4_20261019 | val | 240 | 0.6389 | 0.6458 | 0.6567 | 0.6406 | completed |
| M5_concat_s42_h64_d0.4_20261019 | test | — | — | — | — | — | not_run |
| M5_reference_audio_head_sNA_20261019_01 | val | 240 | 0.4565 | 0.4792 | 0.5054 | 0.4688 | completed |
| M5_reference_audio_head_sNA_20261019_01 | test | 240 | 0.5445 | 0.5750 | 0.5693 | 0.5508 | completed |
| M5_reference_visual_head_sNA_20261019_01 | val | 240 | 0.3033 | 0.3833 | 0.3302 | 0.3594 | completed |
| M5_reference_visual_head_sNA_20261019_01 | test | 240 | 0.2150 | 0.2833 | 0.3312 | 0.2656 | completed |
| M5_fusion_decision_equal_sNA_20261019_01 | val | 240 | 0.4906 | 0.5125 | 0.5450 | 0.5000 | completed |
| M5_fusion_decision_equal_sNA_20261019_01 | test | 240 | 0.5579 | 0.5917 | 0.5843 | 0.5664 | completed |
| M5_fusion_decision_tuned_sNA_20261019_01 | val | 240 | 0.5455 | 0.5958 | 0.6671 | 0.5625 | completed |
| M5_fusion_decision_tuned_sNA_20261019_01 | test | 240 | 0.4741 | 0.5208 | 0.6392 | 0.4883 | completed |
| M5_audio_learned_pilot_s42_20261016_01 | val | 240 | 0.4237 | 0.4500 | 0.4789 | 0.4375 | completed |
| M5_audio_learned_pilot_s42_20261016_01 | test | — | — | — | — | — | not_run |

## Validation-based selections

| Comparison group | Selected run ID | Selection purpose | Reason / controlled factors / limitations |
|---|---|---|---|
| M2_audio_encoders_20261015_00 | M2_encoder_transformer_s42_20261015_00 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261017_01 | M2_encoder_transformer_s42_20261017_01 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261017_02 | M2_encoder_transformer_s42_20261017_02 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261017_03 | M2_encoder_transformer_s42_20261017_03 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261017_04 | M2_encoder_transformer_s42_20261017_04 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261017_05 | M2_encoder_transformer_s42_20261017_05 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261018_01 | M2_encoder_transformer_s42_20261018_01 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261018_02 | M2_encoder_transformer_s42_20261018_02 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261018_03 | M2_encoder_transformer_s42_20261018_03 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261018_04 | M2_encoder_transformer_s7_20261018_04 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2_audio_encoders_20261018_05 | M2_encoder_transformer_s17_20261018_05 | historical within-group encoder selection | Validation Macro-F1; common head family/protocol within group; adaptive warm-start sources disclosed. |
| M2 adaptive overall selection across 33 candidates | **M2_encoder_transformer_s7_20261018_04** | current learned encoder default | Accuracy0.5458333333; Macro-F10.5328369004. Accuracy unchanged vs previous default, Macro-F1+0.0004344239; not a meaningful-generalization claim. |
| M5_fusion_mobilenet_audio176_v1 | [M5_fusion_cross_attention_s42_20261019_01](M5_fusion_cross_attention_s42_20261019_01/run.json); [M5_cross_attention_s43_h64_d0.2_20261019](M5_cross_attention_s43_h64_d0.2_20261019/run.json); [M5_cross_attention_s44_h64_d0.2_20261019](M5_cross_attention_s44_h64_d0.2_20261019/run.json) | primary fusion method, across all seeds | Selected cross_attention by validation mean0.6608; gated test0.6831 is descriptive and did not replace selection. |
| M5_decision_fixed_exports_v1 | M5_fusion_decision_tuned_sNA_20261019_01 | audio probability weight | Validation-only alpha_audio=0.15; class order confirmed afterward without changing values. |

Only actual formal M2 training runs are included: initial3 + 20261017 tuning15 + 20261018 tuning15 =33. Initialization copies, repeated feature exports and synthetic/smoke checks are not extra runs. Test results remain null. Seeds7/17 also change other parameters; do not compute same-config multi-seed mean/std from these trials. Full raw precision is preserved in run.json.

## Synthetic engineering checks

Not included as RAVDESS research results.

## External sample evaluation

No formal external evaluation run is claimed.

## M5 current results and limitations

30 added M5 records =26 trained pipeline configurations/seeds +2 decision configurations +2 imported probability-head references. Existing M2 records are preserved. Means±sample SD are in the [M5 report](shared/M5/M5_RESULTS_20261020.md). Historical features correspond to the initial M2 checkpoint, not the newly promoted round04 default. Synthetic checks are excluded from these real-data counts. No test-score-based reselection was performed.

## WavLM transfer extension — 2026-10-20

Frozen external pretrained backbone; new downstream adapters. Legacy MFCC default remains unchanged. Selected WavLM pipeline: Round05. This is not a same-input architecture-only comparison.

| Run | Extractor / representation | Head | Training | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|---:|---:|
| [M2_encoder_wavlm_s42_20261020_01](M2_encoder_wavlm_s42_20261020_01/run.json) | frozen WavLM + mean → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.675000 | 0.684081 | not_run | not_run |
| [M2_encoder_wavlm_s42_20261020_02](M2_encoder_wavlm_s42_20261020_02/run.json) | frozen WavLM + statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.725000 | 0.727131 | not_run | not_run |
| [M2_encoder_wavlm_s42_20261020_03](M2_encoder_wavlm_s42_20261020_03/run.json) | frozen WavLM + statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.787500 | 0.780654 | not_run | not_run |
| [M2_encoder_wavlm_s42_20261020_04](M2_encoder_wavlm_s42_20261020_04/run.json) | frozen WavLM + attentive_statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.808333 | 0.801674 | not_run | not_run |
| [M2_encoder_wavlm_s42_20261020_05](M2_encoder_wavlm_s42_20261020_05/run.json) | frozen WavLM + attentive_statistics → 128 | MLP128→64→8 / dropout0.4 | frozen backbone; train adapter/head | 0.812500 | 0.806292 | 0.8291666666666667 | 0.8280119634560423 |
