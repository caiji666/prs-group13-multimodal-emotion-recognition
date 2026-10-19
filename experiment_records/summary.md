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

Only actual formal M2 training runs are included: initial3 + 20261017 tuning15 + 20261018 tuning15 =33. Initialization copies, repeated feature exports and synthetic/smoke checks are not extra runs. Test results remain null. Seeds7/17 also change other parameters; do not compute same-config multi-seed mean/std from these trials. Full raw precision is preserved in run.json.

## Synthetic engineering checks

Not included as RAVDESS research results.

## External sample evaluation

No formal external evaluation run is claimed.
