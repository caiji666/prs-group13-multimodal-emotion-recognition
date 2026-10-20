# M3 视觉特征输出说明

## 文件内容

`visual_representation_outputs.npz` 包含 1,440 条原始视频样本：

- train：960 条
- validation：240 条
- test：240 条

请按 `sample_id` 与音频特征配对，不要按数组当前行号直接拼接。

## 类别顺序

`label` 和所有概率字段均使用以下顺序：

```text
0 neutral
1 calm
2 happy
3 sad
4 angry
5 fearful
6 disgust
7 surprised
```

## M5 主要使用字段

### 特征级融合和门控融合

```text
F_visual [1440, 320]
```

最终选中的视频级视觉特征，每条视频对应一个 320 维向量。可用于 Feature-Level Fusion 和基于特征的 Gated Fusion。

### 决策级融合

```text
P_visual_fusion [1440, 8]
```

最终视觉分类器输出的 8 类概率：

- train：按演员分组的 4 折 OOF 概率；
- validation/test：最终锁定模型的预测概率。

可用于 Decision-Level Fusion 和基于概率的 Gated Fusion。不要使用普通的 `P_visual` 代替该字段训练融合模型，因为其 train 部分是最终模型对训练样本的全量预测。

### Cross-Attention

```text
F_visual_deep_sequence [1440, 16, 576]
```

每条视频包含 16 帧深度视觉特征，每帧 576 维，可用于 Cross-Attention Fusion。所有视频均为固定 16 帧，因此：

```text
sequence length = 16
padding mask = 全部 False
```

## 样本信息字段

| 字段 | 形状 | 含义 |
|---|---:|---|
| `sample_id` | `[1440]` | 与音频样本配对的统一 ID |
| `split` | `[1440]` | `train`、`val` 或 `test` |
| `label` | `[1440]` | 0–7 的情绪标签 |
| `class_names` | `[8]` | 标签和概率列的类别顺序 |
| `actor_id` | `[1440]` | 演员编号 1–24 |

## 其他可选字段

三种视觉表示：

```text
F_visual_handcrafted [1440, 320]
F_visual_deep        [1440, 576]
F_visual_combined    [1440, 896]
```

三种表示对应的融合安全概率：

```text
P_visual_handcrafted_fusion [1440, 8]
P_visual_deep_mlp_fusion    [1440, 8]
P_visual_combined_fusion    [1440, 8]
```

逐帧手工视觉序列：

```text
F_visual_handcrafted_sequence [1440, 16, 320]
```

帧时间信息：

```text
frame_idx      [1440, 16]
frame_time_sec [1440, 16]
video_fps      [1440]
```

来源与模型信息：

```text
selected_representation
selected_classifier
deep_source_model
deep_source_checkpoint
```

## M5 接入提示

当前文件保留 M3 的正式字段名称。M5 如需使用内部统一名称，可在读取时映射：

```text
sample_id              -> sample_ids
label                  -> labels
actor_id               -> actors
F_visual               -> pooled
P_visual_fusion        -> probs
F_visual_deep_sequence -> sequence
```

视觉序列固定为 16 帧，M5 可生成：

```text
lengths = 全部 16
padding_mask = 全部 False
```

OOF 折划分和概率来源的详细记录见 `probability_protocol.json`。
