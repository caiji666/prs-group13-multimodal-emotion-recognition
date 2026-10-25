# Group 13 Experiment Summary — v2

> Final evaluation update2026-10-26: all48 M2 records now have test metrics. Weights/configurations and validation-based selection are unchanged. Final model remains M2_encoder_wavlm_refinement_s42_20261021_05 (test Accuracy0.833333, Macro-F10.832127). No test-based model reselection.

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

## Real RAVDESS speech — full-pipeline results

| Run ID | Split | N evaluated | Macro-F1 | Accuracy | Macro-Precision | Macro-Recall | Evaluation status |
|---|---|---|---|---|---|---|---|
| M2_encoder_cnn_s42_20261015_00 | val | 240 | 0.3836 | 0.4250 | 0.4609 | 0.4219 | completed |
| M2_encoder_cnn_s42_20261015_00 | test | 240 | 0.471442 | 0.512500 | 0.489173 | 0.488281 | completed |
| M2_encoder_bilstm_s42_20261015_00 | val | 240 | 0.3593 | 0.4167 | 0.3797 | 0.3906 | completed |
| M2_encoder_bilstm_s42_20261015_00 | test | 240 | 0.492372 | 0.533333 | 0.493690 | 0.500000 | completed |
| M2_encoder_transformer_s42_20261015_00 | val | 240 | 0.4508 | 0.4708 | 0.5393 | 0.4648 | completed |
| M2_encoder_transformer_s42_20261015_00 | test | 240 | 0.546610 | 0.554167 | 0.583612 | 0.539062 | completed |
| M2_encoder_cnn_s42_20261017_01 | val | 240 | 0.4086 | 0.4375 | 0.5073 | 0.4219 | completed |
| M2_encoder_cnn_s42_20261017_01 | test | 240 | 0.537410 | 0.575000 | 0.575969 | 0.550781 | completed |
| M2_encoder_bilstm_s42_20261017_01 | val | 240 | 0.3171 | 0.3875 | 0.3363 | 0.3633 | completed |
| M2_encoder_bilstm_s42_20261017_01 | test | 240 | 0.333974 | 0.400000 | 0.345243 | 0.375000 | completed |
| M2_encoder_transformer_s42_20261017_01 | val | 240 | 0.5204 | 0.5292 | 0.5660 | 0.5234 | completed |
| M2_encoder_transformer_s42_20261017_01 | test | 240 | 0.562137 | 0.587500 | 0.588664 | 0.570312 | completed |
| M2_encoder_cnn_s42_20261017_02 | val | 240 | 0.4282 | 0.4458 | 0.4548 | 0.4414 | completed |
| M2_encoder_cnn_s42_20261017_02 | test | 240 | 0.540859 | 0.562500 | 0.571726 | 0.562500 | completed |
| M2_encoder_bilstm_s42_20261017_02 | val | 240 | 0.3548 | 0.3917 | 0.3794 | 0.3828 | completed |
| M2_encoder_bilstm_s42_20261017_02 | test | 240 | 0.444604 | 0.462500 | 0.452779 | 0.464844 | completed |
| M2_encoder_transformer_s42_20261017_02 | val | 240 | 0.5296 | 0.5458 | 0.5841 | 0.5391 | completed |
| M2_encoder_transformer_s42_20261017_02 | test | 240 | 0.608955 | 0.625000 | 0.633620 | 0.617188 | completed |
| M2_encoder_cnn_s42_20261017_03 | val | 240 | 0.4126 | 0.4333 | 0.4353 | 0.4297 | completed |
| M2_encoder_cnn_s42_20261017_03 | test | 240 | 0.559146 | 0.579167 | 0.592863 | 0.578125 | completed |
| M2_encoder_bilstm_s42_20261017_03 | val | 240 | 0.3914 | 0.4375 | 0.4287 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261017_03 | test | 240 | 0.493692 | 0.520833 | 0.509167 | 0.492188 | completed |
| M2_encoder_transformer_s42_20261017_03 | val | 240 | 0.5263 | 0.5333 | 0.5668 | 0.5352 | completed |
| M2_encoder_transformer_s42_20261017_03 | test | 240 | 0.610592 | 0.629167 | 0.630971 | 0.625000 | completed |
| M2_encoder_cnn_s42_20261017_04 | val | 240 | 0.4241 | 0.4417 | 0.4436 | 0.4414 | completed |
| M2_encoder_cnn_s42_20261017_04 | test | 240 | 0.540114 | 0.562500 | 0.574791 | 0.558594 | completed |
| M2_encoder_bilstm_s42_20261017_04 | val | 240 | 0.3869 | 0.4333 | 0.4027 | 0.4219 | completed |
| M2_encoder_bilstm_s42_20261017_04 | test | 240 | 0.500689 | 0.520833 | 0.520634 | 0.500000 | completed |
| M2_encoder_transformer_s42_20261017_04 | val | 240 | 0.5264 | 0.5375 | 0.5655 | 0.5352 | completed |
| M2_encoder_transformer_s42_20261017_04 | test | 240 | 0.619362 | 0.633333 | 0.641596 | 0.632812 | completed |
| M2_encoder_cnn_s42_20261017_05 | val | 240 | 0.4216 | 0.4417 | 0.4646 | 0.4336 | completed |
| M2_encoder_cnn_s42_20261017_05 | test | 240 | 0.534557 | 0.562500 | 0.559141 | 0.550781 | completed |
| M2_encoder_bilstm_s42_20261017_05 | val | 240 | 0.3818 | 0.4292 | 0.4185 | 0.4102 | completed |
| M2_encoder_bilstm_s42_20261017_05 | test | 240 | 0.489479 | 0.520833 | 0.501223 | 0.492188 | completed |
| M2_encoder_transformer_s42_20261017_05 | val | 240 | 0.5324 | 0.5458 | 0.5756 | 0.5430 | completed |
| M2_encoder_transformer_s42_20261017_05 | test | 240 | 0.620321 | 0.637500 | 0.650922 | 0.628906 | completed |
| M2_encoder_cnn_s42_20261018_01 | val | 240 | 0.4210 | 0.4375 | 0.4655 | 0.4297 | completed |
| M2_encoder_cnn_s42_20261018_01 | test | 240 | 0.526737 | 0.558333 | 0.553728 | 0.546875 | completed |
| M2_encoder_bilstm_s42_20261018_01 | val | 240 | 0.3862 | 0.4333 | 0.4280 | 0.4141 | completed |
| M2_encoder_bilstm_s42_20261018_01 | test | 240 | 0.498646 | 0.529167 | 0.511655 | 0.500000 | completed |
| M2_encoder_transformer_s42_20261018_01 | val | 240 | 0.5324 | 0.5458 | 0.5756 | 0.5430 | completed |
| M2_encoder_transformer_s42_20261018_01 | test | 240 | 0.615488 | 0.633333 | 0.647197 | 0.625000 | completed |
| M2_encoder_cnn_s42_20261018_02 | val | 240 | 0.4234 | 0.4417 | 0.4601 | 0.4375 | completed |
| M2_encoder_cnn_s42_20261018_02 | test | 240 | 0.532773 | 0.562500 | 0.548537 | 0.550781 | completed |
| M2_encoder_bilstm_s42_20261018_02 | val | 240 | 0.3901 | 0.4375 | 0.4258 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261018_02 | test | 240 | 0.506459 | 0.537500 | 0.516346 | 0.507812 | completed |
| M2_encoder_transformer_s42_20261018_02 | val | 240 | 0.5243 | 0.5292 | 0.5604 | 0.5273 | completed |
| M2_encoder_transformer_s42_20261018_02 | test | 240 | 0.621547 | 0.641667 | 0.643641 | 0.632812 | completed |
| M2_encoder_cnn_s42_20261018_03 | val | 240 | 0.4241 | 0.4417 | 0.4660 | 0.4375 | completed |
| M2_encoder_cnn_s42_20261018_03 | test | 240 | 0.541902 | 0.570833 | 0.559569 | 0.558594 | completed |
| M2_encoder_bilstm_s42_20261018_03 | val | 240 | 0.3897 | 0.4375 | 0.4308 | 0.4180 | completed |
| M2_encoder_bilstm_s42_20261018_03 | test | 240 | 0.487728 | 0.520833 | 0.502661 | 0.492188 | completed |
| M2_encoder_transformer_s42_20261018_03 | val | 240 | 0.5292 | 0.5417 | 0.5708 | 0.5391 | completed |
| M2_encoder_transformer_s42_20261018_03 | test | 240 | 0.620321 | 0.637500 | 0.650922 | 0.628906 | completed |
| M2_encoder_cnn_s7_20261018_04 | val | 240 | 0.4202 | 0.4417 | 0.4664 | 0.4336 | completed |
| M2_encoder_cnn_s7_20261018_04 | test | 240 | 0.543284 | 0.570833 | 0.560233 | 0.558594 | completed |
| M2_encoder_bilstm_s7_20261018_04 | val | 240 | 0.3867 | 0.4333 | 0.4270 | 0.4141 | completed |
| M2_encoder_bilstm_s7_20261018_04 | test | 240 | 0.490854 | 0.520833 | 0.503123 | 0.492188 | completed |
| M2_encoder_transformer_s7_20261018_04 | val | 240 | 0.5328 | 0.5458 | 0.5763 | 0.5430 | completed |
| M2_encoder_transformer_s7_20261018_04 | test | 240 | 0.623501 | 0.641667 | 0.653729 | 0.632812 | completed |
| M2_encoder_cnn_s17_20261018_05 | val | 240 | 0.4150 | 0.4333 | 0.4413 | 0.4297 | completed |
| M2_encoder_cnn_s17_20261018_05 | test | 240 | 0.545524 | 0.570833 | 0.566758 | 0.566406 | completed |
| M2_encoder_bilstm_s17_20261018_05 | val | 240 | 0.3878 | 0.4292 | 0.4296 | 0.4141 | completed |
| M2_encoder_bilstm_s17_20261018_05 | test | 240 | 0.496103 | 0.525000 | 0.510187 | 0.496094 | completed |
| M2_encoder_transformer_s17_20261018_05 | val | 240 | 0.5286 | 0.5417 | 0.5737 | 0.5391 | completed |
| M2_encoder_transformer_s17_20261018_05 | test | 240 | 0.621929 | 0.637500 | 0.645215 | 0.628906 | completed |

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

Only actual formal M2 training runs are included: initial3 + 20261017 tuning15 + 20261018 tuning15 =33. Initialization copies, repeated feature exports and synthetic/smoke checks are not extra runs. Final test results were added2026-10-26 without further training or test-based selection. Seeds7/17 also change other parameters; do not compute same-config multi-seed mean/std from these trials. Full raw precision is preserved in run.json.

## Synthetic engineering checks

Not included as RAVDESS research results.

## External sample evaluation

No formal external evaluation run is claimed.

## WavLM transfer extension — 2026-10-20

Frozen external pretrained backbone; new downstream adapters. Legacy MFCC default remains unchanged. Selected WavLM pipeline: Round05. This is not a same-input architecture-only comparison.

| Run | Extractor / representation | Head | Training | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---:|---:|---:|---:|
| [M2_encoder_wavlm_s42_20261020_01](M2_encoder_wavlm_s42_20261020_01/run.json) | frozen WavLM + mean → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.675000 | 0.684081 | 0.720833 | 0.722620 |
| [M2_encoder_wavlm_s42_20261020_02](M2_encoder_wavlm_s42_20261020_02/run.json) | frozen WavLM + statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.725000 | 0.727131 | 0.758333 | 0.756971 |
| [M2_encoder_wavlm_s42_20261020_03](M2_encoder_wavlm_s42_20261020_03/run.json) | frozen WavLM + statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.787500 | 0.780654 | 0.804167 | 0.801471 |
| [M2_encoder_wavlm_s42_20261020_04](M2_encoder_wavlm_s42_20261020_04/run.json) | frozen WavLM + attentive_statistics → 128 | MLP128→64→8 / dropout0.3 | frozen backbone; train adapter/head | 0.808333 | 0.801674 | 0.825000 | 0.824557 |
| [M2_encoder_wavlm_s42_20261020_05](M2_encoder_wavlm_s42_20261020_05/run.json) | frozen WavLM + attentive_statistics → 128 | MLP128→64→8 / dropout0.4 | frozen backbone; train adapter/head | 0.812500 | 0.806292 | 0.829167 | 0.828012 |

## WavLM continuation — ten experiments, 2026-10-21

Development used only training/validation data; final tests were subsequently completed2026-10-26 without further tuning. Backbone frozen. Current research validation winner is continuation05, distinct from original Round05. Existing deployed/default models and handoff remain unchanged.

| Run | Parent | Representation | Classifier | Freeze prefixes | Val Accuracy | Val Macro-F1 | Test Accuracy / Macro-F1 |
|---|---|---|---|---|---:|---:|---|
| [M2_encoder_wavlm_refinement_s42_20261021_01](M2_encoder_wavlm_refinement_s42_20261021_01/run.json) | M2_encoder_wavlm_s42_20261020_05 | frozen WavLM6 + head_only, 128d | MLP, dropout0.3 | ['projection', 'attention', 'statistics'] | 0.808333 | 0.802598 | 0.829167 / 0.828012 |
| [M2_encoder_wavlm_refinement_s42_20261021_02](M2_encoder_wavlm_refinement_s42_20261021_02/run.json) | M2_encoder_wavlm_s42_20261020_05 | frozen WavLM6 + freeze_projection, 128d | MLP, dropout0.3 | ['projection'] | 0.800000 | 0.791611 | 0.829167 / 0.828012 |
| [M2_encoder_wavlm_refinement_s42_20261021_03](M2_encoder_wavlm_refinement_s42_20261021_03/run.json) | M2_encoder_wavlm_s42_20261020_05 | frozen WavLM6 + gentle_full_adapter, 128d | MLP, dropout0.25 | [] | 0.820833 | 0.810180 | 0.829167 / 0.828012 |
| [M2_encoder_wavlm_refinement_s42_20261021_04](M2_encoder_wavlm_refinement_s42_20261021_04/run.json) | M2_encoder_wavlm_refinement_s42_20261021_03 | frozen WavLM6 + residual_frame, 128d | MLP, dropout0.3 | [] | 0.820833 | 0.809716 | 0.825000 / 0.821541 |
| [M2_encoder_wavlm_refinement_s42_20261021_05](M2_encoder_wavlm_refinement_s42_20261021_05/run.json) | M2_encoder_wavlm_refinement_s42_20261021_03 | frozen WavLM6 + temporal_context, 128d | MLP, dropout0.3 | [] | 0.841667 | 0.831218 | 0.833333 / 0.832127 |
| [M2_encoder_wavlm_refinement_s42_20261021_06](M2_encoder_wavlm_refinement_s42_20261021_06/run.json) | M2_encoder_wavlm_refinement_s42_20261021_05 | frozen WavLM6 + deeper_head, 128d | MLP, dropout0.35 | [] | 0.820833 | 0.811222 | 0.833333 / 0.832127 |
| [M2_encoder_wavlm_refinement_s42_20261021_07](M2_encoder_wavlm_refinement_s42_20261021_07/run.json) | M2_encoder_wavlm_refinement_s42_20261021_05 | frozen WavLM6 + freeze_head_refine_sequence, 128d | MLP, dropout0.25 | ['head'] | 0.816667 | 0.804486 | 0.833333 / 0.832127 |
| [M2_encoder_wavlm_refinement_s42_20261021_08](M2_encoder_wavlm_refinement_s42_20261021_08/run.json) | M2_encoder_wavlm_refinement_s42_20261021_05 | frozen WavLM6 + feature_noise_mixup, 128d | MLP, dropout0.35 | [] | 0.808333 | 0.799298 | 0.833333 / 0.832127 |
| [M2_encoder_wavlm_refinement_s42_20261021_09](M2_encoder_wavlm_refinement_s42_20261021_09/run.json) | M2_encoder_wavlm_refinement_s42_20261021_05 | frozen WavLM6 + focal_classification, 128d | MLP, dropout0.25 | [] | 0.808333 | 0.794674 | 0.833333 / 0.832127 |
| [M2_encoder_wavlm_refinement_s42_20261021_10](M2_encoder_wavlm_refinement_s42_20261021_10/run.json) | M2_encoder_wavlm_refinement_s42_20261021_05 | frozen WavLM6 + final_conservative, 128d | MLP, dropout0.2 | [] | 0.816667 | 0.806847 | 0.816667 / 0.811718 |
