# Group 13 多模态情绪识别融合模块

根据上一级目录 `Project_Proposal_Group13.docx` 第 4 节和第 5 节实现。这个项目让你在队友的音频、视觉编码器完成前，先开发、验证融合模块。输入是冻结编码器导出的特征和八分类概率；输出是融合模型、预测概率和评估结果。

**目前的初步运行使用合成特征，只有工程验证意义。合成数据的高分由生成方式决定，不能写成 RAVDESS 实验结果，也不能用于决定最终融合方案。** 原始音视频提取、真实编码器训练、外部视频演示仍需接入队友的模块。

## 2026-10-19 类别顺序已确认

用户已向队友确认：视觉概率列顺序与团队八类顺序一致。输入文件及已存概率核对未变化，决策结果无需重跑：等权测试Macro-F1=0.5579，验证调权（音频0.15）测试Macro-F1=0.4741。当前已移除未确认类别顺序标记，原测试前锁定记录保持不变；确认依据见结果目录class_order_confirmation.json。后续重点核对视觉原分类头与特征/概率导出的一致性，不能仅凭低分判断有错误。

## 2026-10-19 最终测试与暂定决策融合（确认前历史记录）

按用户新授权完成锁定21个模型的测试，未重训或用测试重选配置。测试Macro-F1：concat0.6480±0.0126、gated0.6831±0.0243、cross_attention0.5749±0.0072。Cross-attention是测试前按验证均值预选的主方案，其测试优势未保持；不按测试排名改写选择。

视觉概率顺序尚未确认，用户授权按约定顺序试跑。暂定等权决策测试F1=0.5579，验证集选音频权重0.15后测试F1=0.4741，均须保留未确认映射标记。完整记录见 [测试与暂定决策融合结果](results/m5_final_evaluation_20261019/测试与暂定决策融合结果.md)。17项测试通过，无后台任务。测试集现在已经查看，后续改方案应记录为探索。

## 2026-10-19 按计划完成的验证研究（测试前历史记录）

已完成3个seed的7种方法比较、4组定向调参及解释分析，共25组配置/seed结果（22次新增训练）。验证Macro-F1：音频控制0.4682±0.0163、视觉控制0.5178±0.0172、拼接0.6392±0.0154、门控0.6011±0.0109、cross-attention0.6608±0.0163。均为固定演员划分的验证结果，未测试；决策融合仍待确认。

完整记录、错误互补、模态干预、gate与attention对照见 [实验结果与解释分析](results/m5_validation_study_20261019/实验结果与解释分析.md)。均匀attention与原模型表现接近，因此不能将当前提升解释成已经学会精细时间对齐。单模态数值指同协议重训MLP控制，不是全组最终上游基线。

新增 `fusion.study`、`fusion.study_diagnostics`、`fusion.study_report` 可重现研究；14项测试通过。原始输入保持冻结，测试集继续保留，无后台训练。

## 2026-10-19 验证专用融合入口（历史记录）

已完成真实首轮：concat val macro-F1=0.6485、gated=0.6043、cross_attention=0.6674，均为seed42；没有测试评价或决策融合。详细结果见 [首轮记录](results/m5_feature_pilot_20261019_s42/初步实验结果.md)。

```bash
python -m fusion.feature_pilot --output results/new_feature_pilot_s42 --epochs 30 --seed 42
```

入口默认从仓库features/读取176D音频、音频序列和MobileNet视觉特征，使用正式actor CSV，仅解码train/val数值，不加载上游概率。缓存为独立feature-only接口，不要用旧fusion train加载。本轮11项测试通过；其他seed、单模态控制和解释分析尚未运行。默认将视觉16个token视为有效，此假设待上游确认。

## 2026-10-19 MobileNet 视觉输出

新增 `features/mobilenetv3_small_features.npz`，包含 [1440,16,576] 视觉序列、[1440,576] pooled 特征和 [1440,8] 概率。全部 ID、标签、正式划分及数值检查通过，可以准备 learned 视觉融合候选实验；概率列顺序、16 帧有效性、模型训练来源及最终视觉选择仍需上游确认。详细检查见 [实验规划文档](实验规划与当前代码说明.md) 的 10 月 19 日章节。本次没有训练。

## 2026-10-18 新交付审核（历史记录）

已收到 `features/` 下最终 176D 音频、M2 音频序列和 320D 手工视觉，1440 条样本按正式 ID 配对与划分检查通过。`video_dl` 实际是人脸图像，还没有 M4 深度特征。音频 CSV 的七段 sample_id / 五段 trial_id 必须先恢复为六段标准 ID；融合主入口还需验证专用模式。可准备手工视觉版本的真实融合，尚未启动训练。详细证据和实验范围见 [实验规划与当前代码说明](实验规划与当前代码说明.md) 中 10 月 18 日检查章节。

## 2026-10-16 真实音频进度（历史记录）

已接入 M2 learned-only 音频特征并完成短时验证，尚无真实视觉融合结果。请先查看 [初测与续跑说明](results/m2_audio_pilot_20261016/初测记录与续跑说明.md)。全组正式分组为 `configs/actor_splits.csv`，运行时显式传 `--split configs/actors_m2.json`。本次未计算测试指标；`fusion.audio_pilot` 是验证专用入口。

## 已实现的方法

| 方法 | 输入 | 实现 |
|---|---|---|
| audio_only / visual_only | 各自八分类概率 | 直接评价队友的分类器输出 |
| decision_equal | 两路概率 | 音频、视觉各 0.5 |
| decision_tuned | 两路概率 | 验证集搜索音频权重 0～1，步长 0.05 |
| concat | 两路 pooled 特征 | 各自训练集标准化 → 拼接 → MLP |
| gated | 两路 pooled 特征 | 投影至同一维度 → tanh → 每个样本、每个维度的 sigmoid gate → MLP |
| cross_attention | 两路 sequence + 有效长度 | audio Q、visual K/V → audio residual → LayerNorm → masked mean pooling → MLP |

这是四条替代融合路径，编码器没有串联。缓存特征不参与梯度计算，只有融合模块训练。Cross-attention 支持不同帧数和不同特征维度，屏蔽视觉 padding，并在音频池化时排除 padding。它依赖队友输出保留时间信息的序列表示；目前没有额外的时间戳位置编码或逐帧音画对齐约束。

## 现在就运行

在终端进入本项目文件夹。已验证本机 `mr` conda 环境能直接运行，无需安装依赖：

```bash
conda activate mr
python -m fusion demo --output results/my_first_demo --epochs 30
python -m unittest discover -s tests -v
```

如果 conda 尚未初始化，可以使用已验证的 Python：

```bash
/Users/jiang/miniconda3/envs/mr/bin/python -m fusion demo --output results/my_first_demo --epochs 30
```

每次用新的 `--output` 路径保留实验记录。默认使用 CPU；`--device mps` / `cuda` 为可选，尚未在本项目验证。若设备不支持确定性操作会报错，初测建议 CPU。

在其他电脑新建环境后安装：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 演员划分和实验纪律

正式共享文件为 [configs/actor_splits.csv](configs/actor_splits.csv)：train=[1,4,5,6,7,10,11,12,13,15,16,18,19,20,23,24]；val=[9,14,21,22]；test=[2,3,8,17]。禁止再次随机分组。完整 speech trial 为 960/240/240 条。`configs/actors.json` 保留合成开发的历史默认值，CLI 默认已改为正式 actors_m2.json；真实数据命令仍建议显式传 `--split configs/actors_m2.json`。

音频、视觉的编码器训练、模型选择以及融合实验必须使用这份共享划分。fusion 只能检查文件中的演员分组，无法证明上游编码器没有看过测试演员。

- 标准化的均值、标准差只从训练演员计算。序列统计不包含 padding。
- 神经网络用验证 macro-F1 选择 checkpoint，默认最多 30 epochs、patience=8；不按测试分数选择 epoch。
- 决策融合的权重只搜索验证集。
- 测试集输出所有方法的指标供最终比较，`selected_by_validation` 按验证分数选择；不要看过测试结果后继续调参。
- 提供固定八类 macro-F1、accuracy、逐类 precision/recall/F1、原始计数混淆矩阵。
- 多 seed 重复训练不改变正式演员划分；其他演员划分只能作为另行确认的扩展研究。本项目未完成重复真实演员划分或统计显著性检验。

## 队友交接格式

团队 Sample ID 规则、参考生成函数、必需元数据及融合验收清单见 [实验规划与当前代码说明](实验规划与当前代码说明.md)。跨 M1–M5 保存 sample_id、split、label 与 features，预测保留类别顺序；按 ID 配对且每模态内部 ID 唯一。当前表格中 sample_ids/labels 等复数字段是对应的批量接口命名。

音频与视觉各交一个 `.npz` 文件，统一类别顺序：

```text
0 neutral, 1 calm, 2 happy, 3 sad, 4 angry, 5 fearful, 6 disgust, 7 surprise
```

| 字段 | 形状 / 类型 | 说明 |
|---|---|---|
| sample_ids | `[N]` Unicode 字符串 | 不含 modality 的六段 trial ID |
| labels | `[N]` int64 | 0～7，RAVDESS emotion 编码减 1 |
| actors | `[N]` int64 | 1～24 |
| class_names | `[8]` Unicode 字符串 | 必须按上面的顺序 |
| pooled | `[N,D]` float32 | 当前选定的完整 pooled 表示 |
| probs | `[N,8]` float32 | 概率，非 logits，每行和为 1 |
| sequence | `[N,T,Ds]` float32，可选 | 选定 learned encoder 的序列，后部补零 |
| lengths | `[N]` int64，可选 | 每个样本有效时间步数，必须与 sequence 一起提供 |

例如 `03-01-06-01-02-01-12.wav` 和 `02-01-06-01-02-01-12.mp4` 都对应 `01-06-01-02-01-12`。只去掉第一段 modality，不能去掉 intensity、statement 或 repetition。音频与视觉文件行顺序可不同，程序按 ID 重排；不允许悄悄丢弃缺失配对。speech 的 channel 必须是 01；song 被拒绝。

音频 pooled 应是选定 learned pooled 与 handcrafted 的最终拼接表示。视觉 pooled 应是验证集选出的 A、B 或 A+B 表示。用于 cross-attention 的 sequence 应分别来自选定 learned encoders，不需要和 pooled 维度相同。即使视觉最终选出 DWT-SVM，若要测试 cross-attention，仍需要保留 learned sequence。

可复制 `examples/export_features.py` 的 `export_features` 函数到队友的导出代码中。PyTorch 输出先 `.detach().cpu().numpy()`；导出时模型用 `eval()` 和 `torch.inference_mode()`。SVM 的 `predict_proba` 列要按 `classes_` 重新排列。

建议额外交接：encoder 名称和 checkpoint、特征维度、预处理与序列采样方式、帧时间戳、训练演员列表、验证选择记录。这些元数据由团队记录；当前 NPZ 校验不替代上游实验审核。

## 接入真实特征

```bash
python -m fusion validate --audio data/real/audio.npz --visual data/real/visual.npz --split configs/actors_m2.json
python -m fusion train --audio data/real/audio.npz --visual data/real/visual.npz --split configs/actors_m2.json --output results/real_v1 --epochs 50
```

队友暂时只有 pooled 和 probs 时，可以先运行三条主要路径：

```bash
python -m fusion train --audio data/real/audio.npz --visual data/real/visual.npz --split configs/actors_m2.json --output results/real_pooled_v1 --methods decision concat gated
```

## 复用模型预测

```bash
python -m fusion predict --checkpoint results/synthetic_seed42/checkpoints/gated.pt --audio data/synthetic/audio.npz --visual data/synthetic/visual.npz --output results/demo_predictions.json
```

决策融合使用 `decision_equal.json` 或 `decision_tuned.json` checkpoint。预测 JSON 包含 emotion、confidence 和完整八类概率。接口目前使用带 RAVDESS ID 和标签的特征文件，适合离线演示；原始外部视频的无标签上传界面未实现。Confidence 是模型概率，尚未做校准。合成 checkpoint 只用于演示，真实特征必须重新训练。

## 文件与结果

- `fusion/data.py`：配对、校验、演员划分、标准化。
- `fusion/models.py`：三种可训练融合模型。
- `fusion/experiment.py`：训练、模型选择、评估、checkpoint 和推理。
- `fusion/demo.py`：24 位模拟演员、1440 条 speech trial 结构的合成 fixture。
- `results/synthetic_seed42/`：已跑的初测。
- `metrics.json`：完整指标、环境、配置和合成标识。
- `summary.csv`：方法对照；`history.json`：训练曲线数值。
- `test_predictions.csv`：逐样本预测；`split_manifest.csv`：逐样本演员划分。
- `confusion_*.png`：混淆矩阵；`checkpoints/`：最佳权重和训练集标准化参数。

测试覆盖完整 trial 配对、乱序配对、缺失配对拒绝、演员泄漏拒绝、训练集统计隔离、非法概率拒绝、padding 不变性，以及所有融合方法 checkpoint 的预测复现。

实现 cross-attention 的 padding 行为参考 [PyTorch MultiheadAttention 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html)。本机实际验证版本记录在每次实验的 `metrics.json`。

## 旧特征结构实验（2026-10-20）

暂不接WavLM，已完成9种结构×3seed的27次训练，仅看验证集。详见[结果解读](results/m5_structure_study_20261020/结果解读.md)与[方法及复跑命令](results/m5_structure_study_20261020/方法与复跑说明.md)。新增structure_models、structure_study与structure_report模块，既有模型与成绩保留。

## 文献候选实验（2026-10-20）

已完成LMF rank4、SupCon及对应控制15组训练，27个SVM/核融合候选，以及gate覆盖诊断；均为旧特征验证集实验。见[结果与解释](results/m5_literature_study_20261020/实验结果与解释.md)。新增literature_models、literature_study、literature_report模块；历史结果保留。

## 最终交付重跑（2026-10-27）

当前主输入WavLM-only128D＋handcrafted视觉320D，完整重跑130条记录。见[最终结果与解释](results/m5_final_delivery_20261027/最终特征实验结果与解释.md)与[逐配置总表](results/m5_final_delivery_20261027/summary.md)。旧命令和结果属于历史特征版本；本轮使用fusion.final_study显式指定新来源，先锁定验证再评价test。

## 本地视频推理与解释 UI（2026-10-27）

已新增[本地演示系统](../demo/README.md)。在项目根目录执行 `demo/.venv/bin/python -m demo.server --port 8765`，打开 `http://127.0.0.1:8765`。原始视频可提取 WavLM、M1手工音频及组合音频，以及 MediaPipe＋DWT视觉；仅开放有匹配checkpoint的融合组合。解释包括决策加权分解、64维gate、attention关联图、音频残差修正和固定目标输入敏感性。

深度视觉原权重与最终视觉SVM尚未交付；原始视频使用M5视觉MLP及单独验证调权协议，不能套用论文中SVM决策融合的分数。验证回放可使用三种现有视觉表示及交付概率。10项回归检查及合成视频工程链路通过；尚无新的外部视频情绪准确率评价。历史模型、选择记录和test结果保持原样。

## Published M5 experiment records

The current results are available in the repository under [experiment_records](../experiment_records/summary.md), using the shared v2 format. The [M5 report](../experiment_records/shared/M5/M5_RESULTS_20261020.md) includes multi-seed controls, locked test evaluation, confirmed-order decision fusion and interpretation diagnostics. Per-run notes disclose historical input versions, missing upstream metadata and test exposure.

Run `python ../scripts/validate_m5_records.py --root ..` to verify predictions and scores from this directory. The new feature-only training, study and diagnostics modules require separately obtained historical features; generated local results are not uploaded wholesale.
