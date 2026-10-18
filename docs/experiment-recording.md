# M1–M5 统一实验记录规范 v2

日期：2026-10-18。根据团队分工图、已约定的 actor split / sample ID 和当前融合程序输出设计。本文件是建议采用的团队记录规范，不是新增的课程要求。

本版把每次实验明确表示为 **特征提取 → 表示组合 → 可选多模态融合 → 分类头**。总表分别展示流水线配置和结果，通过 run_id 关联。分类指标属于完整流水线；不把分类头隐含在 encoder 名称里，也不假设特征提取器和分类头一定分开训练。

## 1. 大家怎么用

每次实验按“一个完整流水线配置 × 一个训练 seed”记录，分配唯一 `run_id`。同一已训练流水线的 val/test 结果放在同一记录中；更换特征提取器、输入表示、分类头、训练方式、超参数、上游 checkpoint 或重新训练，都新建记录，不覆盖旧实验。

建议目录如下，`experiment_records/` 可以进 Git；数据、权重和大特征继续放独立存储。

```text
experiment_records/
  summary.md                         # 复制全组总表模板
  M2_encoder_transformer_s42_20261018_01/
    run.json                         # 配置、来源、指标：复制 run.template.json
    notes.md                         # 目的、发现、解释：复制 experiment.template.md
    predictions_val.csv              # 程序导出，格式见第 6 节
    predictions_test.csv             # 最终测试完成后才生成
    history.csv                      # 神经网络训练时保存；传统模型可不提供
    figures/                         # 混淆矩阵、训练曲线等
```

模板：[全组总表](templates/summary.template.md)、[实验说明](templates/experiment.template.md)、[配置与结果 JSON](templates/run.template.json)。模板不含虚构实验分数。

最小操作：复制一次实验的 JSON 和说明模板，填好配置与结果，在全组总表加一行。JSON 是结构化原始记录，总表只摘录用于比较的关键值，不独立手改一套分数。已有程序导出的原始结果可以保留，由记录中的 `artifacts` 指向；不要求大家重写训练代码。

## 2. 各成员应该记录哪些实验

| 成员 | stage | 至少保留的比较 | 必须注明的依赖或配置 |
|---|---|---|---|
| M1 Audio Handcrafted | `representation` | handcrafted-only / M2 selected learned-only / handcrafted+learned | ZCR、RMS/Energy、FFT/Spectral、DWT 的具体特征与参数；分类器；组合时引用 M2 run_id |
| M2 Audio Deep Learning | `encoder` | 1D-CNN / LSTM 或 BiLSTM / Transformer | MFCC 参数、时序处理、各模型配置；用验证 Macro-F1 选出的 learned encoder |
| M3 Visual Handcrafted | `representation` | DWT-SVM / M4 selected learned-only / DWT+learned+MLP | DWT wavelet/level、统计和 edge 特征、帧数与 pooling；组合时引用 M4 run_id |
| M4 Visual Deep Learning | `encoder` | MobileNetV3-Small+mean / ResNet18+mean / ResNet18+BiLSTM | 采帧、face crop/alignment、预训练来源、冻结/微调层；验证集选出的 learned encoder |
| M5 Multimodal Fusion | `fusion` | audio-only / visual-only / decision_equal / decision_tuned / concat / gated / cross_attention | 引用最终 audio、visual 的 run_id 与特征版本；是否冻结；gate/attention 参数 |

M1/M3 的 learned-only 对照可直接引用 M2/M4 已有 run，而不是假装又训练了一次。若更换 classifier、重新训练或重新导出特征，则创建新 run，并记录 parent run。M2/M4 选出“最佳 learned encoder”，M1/M3 再比较三种完整表示并确定交给 M5 的配置，两个选择步骤要分别记录。

表格忠实保留分工，但不强制各方法采用同一种分类器。比如 DWT-SVM 与 learned+MLP 的差异同时包含表示和分类器差异，报告不能直接归因于“特征更好”。要证明单独某个因素的作用，需要额外控制变量实验。

### 特征提取与分类头必须分别记录

| 层次 | 必填内容 | 示例（仅结构示意，不代表实际实验配置） |
|---|---|---|
| 特征提取器 | 方法、输入、参数、预训练来源、sequence/pooled 输出形状、是否参与本次训练 | MFCC → Transformer → `[N,T,128]` → mean → `[N,128]` |
| 表示组合 | 特征来源、pooling、标准化、组合方式、各分量及最终维度 | handcrafted `[N,Dh]` + learned `[N,128]` → `[N,Dh+128]` |
| 分类头 | 类型、完整层宽/激活/dropout 或 SVM 参数、概率输出方式 | MLP `128→64→8`；或 RBF-SVM，记录 C/gamma 与校准 |
| 训练关系 | 手工特征后训练分类器 / frozen encoder 后训练分类器 / joint training / partial fine-tuning | 冻结上游 Transformer，只训练新的 MLP |
| 结果 | 完整流水线的 val/test 指标；选中 checkpoint | `Transformer pooled → MLP` 的 Macro-F1，而不是“Transformer 特征的 F1” |

M1/M3 同时使用两种特征时，`feature_extractors` 填两项，`representation.components` 引用对应 component_id。M4 的 ResNet18+BiLSTM 中，BiLSTM 属于时序特征提取器，MLP 是分类头。即使代码将它们写在同一模型类中，记录时仍按功能拆开。

M5 的 concat/gated/cross-attention 记录 `pipeline.fusion` 和后续分类头；decision fusion 没有新 MLP 时，`classifier.type="none"`、`classifier.output_dim=null`，并在 fusion 的 `input_source_runs` 中引用已有 audio/visual 分类流水线。已有上游分类器结构可以从来源记录追溯，不虚构新的分类头。

用于判断“特征提取器哪个好”时，尽量固定分类头类型、hidden layers、激活、dropout、训练规则和调参预算。输入维度不同，首层参数量可以不同，需如实记录，不能声称分类头参数完全相同。比较分类头时应固定同一份特征及 hash；若 joint training 导致特征也改变，就不能解释成只改变分类头的实验。

## 3. 每次实验的身份和来源

推荐 ID：`M2_encoder_transformer_s42_20261018_01`。最后两位用于同一天同模型的不同运行。seed 不适用时写 `sNA`，并解释是否仍存在随机划分、训练或概率校准步骤。

| 字段 | 填什么 |
|---|---|
| `member` / `stage` / `pipeline_name` | 谁负责、比较哪一阶段、完整流水线名称 |
| `comparison_group` | 同一公平比较的候选组，如 `audio_encoders_v1`；协议/输入集合变化应建新组 |
| `dataset_id` / `data_kind` | 如 `ravdess_speech_v1` + `real`；合成用 `synthetic`，外部样本用 `external` |
| `git_commit` / `working_tree_dirty` / `command` | 完整代码版本、是否有未提交修改、实际运行命令；有修改时保留 patch 并填写 `code_patch_ref` |
| `seed` / `environment` | seed、Python/库版本、设备；建议保存 environment 文件 |
| `split_ref` / `split_sha256` | 共享 actor CSV 路径及文件内容 SHA-256，防止同名但内容不同 |
| `input_manifest_ref` / `input_manifest_sha256` | 每条样本的 sample_id、actor_id、split、label；hash 标识实际纳入的数据 |
| `source_runs` / `input_artifacts` | 上游 run_id、checkpoint/特征位置与 SHA-256，特别说明 learned-only 或 combined |
| `pipeline.feature_extractors` | 每个特征提取器的 component_id、方法/参数、checkpoint、来源、输出形状和本次 trainable 状态 |
| `pipeline.representation` | handcrafted / learned / combined，组成 component_id、pooling/标准化、组合方式和输出形状 |
| `pipeline.fusion` | 是否执行跨模态融合、方法、参数、来源 run 与输出形状；单模态填 applied=false |
| `pipeline.classifier` | MLP/SVM/其他分类器或 none；输入维度、结构、dropout/激活、概率方式、checkpoint |
| `preprocessing` / `training` | 实际预处理、训练关系、optimizer、lr、batch、class weighting；组件不同 lr 等填 component_parameter_groups |
| `selection` | 用哪个指标、在哪个候选组内选择、最佳 epoch、是否锁定配置，以及选择理由 |
| `counts` / `excluded_samples` | 原始/实际样本数；缺脸、坏文件等排除的 ID、原因和处理方式 |

不要只写“默认参数”“best model”或“用队友给的特征”。至少要能找回具体输入和模型。

所有 `*_ref` 优先用仓库根目录相对路径或团队可访问的存储链接，避免只写个人电脑绝对路径。`input_artifacts` 每项填写 `role`、`source_run_id`、`ref`、`sha256`；`feature_exports` 每项填写 `role`、`ref`、`sha256`、`pooled_shape`、`sequence_shape`、`lengths_shape` 和 `mask_convention`。不提供 sequence 时对应项填 null，并解释限制。共享的统计数据和预测文件要避免包含私人路径或凭据。

`feature_extractors` 可以复制为多项；`kind` 推荐 `handcrafted` 或 `learned`。`trainable_in_this_run` 表示本次是否更新该组件参数，不能用它表示“曾经训练过”。手工特征没有学习参数，填 false，但 scaler 仍只能在 train 拟合。MLP 的 `architecture` 写完整结构，`hidden_dims` 只写隐藏层，`input_dim` 单独记录；SVM 的 hidden_dims、activation、dropout 等不适用字段留 null，把 kernel/C/gamma 等写在 parameters。

`training.regime` 推荐 `handcrafted_then_classifier / frozen_encoder_then_classifier / joint_training / partial_finetune / probability_fusion`。组件独立学习率等在 `component_parameter_groups` 每项记录 `component_id`（或 classifier/fusion）、optimizer、learning_rate、weight_decay。使用两阶段训练时把阶段顺序和各阶段参数记录在 `training.other_parameters.phases`，不要只填最后一阶段。

## 4. 全组统一的评价口径

- 主实验使用 `multimodal_fusion/configs/actor_splits.csv`：train 16 人/960 条，val 4 人/240 条，test 4 人/240 条。禁止重新随机分样本。真实融合命令显式传 `--split configs/actors_m2.json`。
- 固定类别：`0 neutral, 1 calm, 2 happy, 3 sad, 4 angry, 5 fearful, 6 disgust, 7 surprise`。`surprised` 只能作名称别名，不得改变概率列位置。
- 汇总指标：`macro_f1` 为主，辅以 `accuracy`、`macro_precision`、`macro_recall`。Macro 始终平均固定八类；分母为 0 的 precision/recall/F1 记 0（`zero_division=0`）。不以 weighted-F1 或 micro-F1 替代 Macro-F1。
- 文件内所有分数用 **0–1 小数**，如 `0.4532`；正文/PPT 可显示 `45.32%`。从 `0.4532` 到 `0.4832` 写“提高 3.00 个百分点”，不要写成“提高 3%”。
- 保留程序原始精度，总表显示四位小数，不能先逐类四舍五入再算 Macro-F1。
- `per_class` 提供每类 precision、recall、F1、support。混淆矩阵必须保存 **8×8 原始计数**，行=真实类别，列=预测类别，顺序固定；归一化图可以附加。
- 原始计数矩阵之和必须等于实际评估数；逐类 support 等于各行和；accuracy 等于对角线和/总数。分数从相同预测文件计算。
- 正式完整 val/test 中 neutral 各 16 条，其他七类各 32 条。若实际数量不同，说明原因；不能静默丢掉难样本或配对失败样本。缺脸时的 fallback 也要固定并记录。
- 不同方法主比较必须使用相同真实样本 ID 集。若某方法无法覆盖全部样本，先列覆盖率与缺失原因；如做共同子集对比，另设 comparison_group，并为所有对照在同一子集重算，不能直接混用原分数。

### 验证集与测试集

开发阶段记录 val，test 尚未运行写 JSON `null`、表格 `—`，并用 `test_status=not_run`；**0 分不是“未测”**。失败、未做、上游特征未就绪也应明确标记，不能猜分数。

用 val Macro-F1 选择 encoder、表示、epoch、超参数、融合权重和最终方法。建议平分时采用事先约定的更简单模型；decision 权重平分时优先更接近 0.5 的值。选择范围必须写在 comparison_group 中。

模型和比较计划锁定后，再对事先列明的候选/对照做最终 test 评估。同一锁定 checkpoint 的 test 结果补到原 run，不因测试分数重选模型；如果测试后又改方案，应记录为后续探索并披露测试集已被查看。当前 fusion `train` 入口会输出 test，因此不要拿它做反复试参的验证专用入口。

同一固定 actor split 可选跑多个 seed（建议资源允许时用 42/43/44）。每个 seed 单独记录，汇总 mean±sample std（ddof=1）及 n；n=1 不报 std。不能只挑最好 seed，不能把初始化 seed 的重复当成不同演员划分，也不能据此声称统计显著。更换 actor split 的扩展研究另建协议。

## 5. 最低结果材料

| 材料 | 何时需要 | 用途 |
|---|---|---|
| `run.json` | 每个 run | 身份、提取器/表示/融合/分类头的分项配置、训练关系、来源、val/test 四项汇总指标；完成评估后附逐类指标与混淆矩阵 |
| `notes.md` | 每个 run，3–6 句即可 | 实验目的、关键发现、解释、局限和下一步 |
| 逐样本 predictions | 每个已评价 split | 重算指标、检查标签对齐、M5 分析互补性；重复引用 baseline 不复制成独立实验 |
| 训练 history | 可训练神经网络 | 每 epoch 的 train loss、val Macro-F1，核对最佳 epoch 与过拟合 |
| checkpoint / 特征交接说明 | 入选并交给下游的配置 | 位置、hash、输入/输出维度、预处理、类别顺序、序列长度/mask 约定 |

`run.template.json` 中的 `null` 都是待填值；不适用时可留 null，但在 notes 解释。`test_status` 取 `not_run / completed / failed`；`run_status` 取 `planned / completed / failed / blocked`，此处 completed 指预定工作完成，验证专用 run 完成时 test 仍可为 not_run。`selection.selected_for_next_stage` 仅表示该候选组的选择，不代表全组最佳。

测试完成后，把 `metrics_by_split.val` 的结构复制到 `test` 并填测试值；不能把验证分数复制过去。外部评估使用单独 run / dataset_id，并将对应评价分组记为 `external`，而不是改写 RAVDESS 的 test 结果。

JSON `schema_version=2.0`，不兼容把全部模型参数混写在 v1 `model_config` 中的习惯。已有 v1 记录保留原文件，按实际配置迁移字段；不能从模型名字猜分类头。`metrics_scope=full_pipeline` 明确分类分数评价完整流水线。提取耗时、特征形状等可另附诊断，不能当作分类性能。

## 6. 程序导出格式

逐样本 CSV 推荐表头（这是接口说明，可由各自程序生成，不要求手工填写）：

```text
run_id,dataset_id,split,sample_id,actor_id,y_true,y_pred,p_neutral,p_calm,p_happy,p_sad,p_angry,p_fearful,p_disgust,p_surprise
```

每个文件内 `(run_id,dataset_id,split,sample_id)` 唯一；标签为 0–7 整数，概率为有限数且在 [0,1] 内，行和允许浮点误差（建议绝对误差 ≤1e-5）。`y_pred` 取固定顺序 argmax，平分取最小类别索引。若原分类器使用不同决策规则，应注明，并同时保留原输出供核对。SVM 概率列按 `classes_` 重排；如需要校准，只能在训练集内部拟合，记录方法，不能用 test 校准。不具备概率的模型可以先报分类指标，但交给 decision fusion 前必须提供可靠导出的概率。

history 表头推荐：`epoch,train_loss,val_macro_f1`。不要为 SVM 编造 epoch/loss；其调参候选与 val 结果可另附搜索表。

输入特征仍遵循原有 NPZ 交接协议，这次规范不修改它。每个 selected run 的交接说明注明 pooled、sequence、lengths 的形状和含义；仅有 pooled 的临时交接必须标明不能支持 cross-attention。音视频配对继续使用六段 sample_id，与预测文件中的 ID 相同。

### 兼容当前 M5 输出

不改已有程序时，可以从现有文件提取这些字段：

| 当前输出 | 统一记录 |
|---|---|
| `metrics.json` → `models.<method>.val/test.accuracy`、`macro_f1` | 对应 `metrics_by_split.val/test` 的同名字段 |
| `per_class['macro avg'].precision/recall` | `macro_precision` / `macro_recall` |
| `per_class.<emotion>['f1-score']` | 统一 `per_class` 对应项的 `f1` |
| `per_class.<emotion>.support` | 核对为整数后填 `support` |
| `confusion_matrix` | `confusion_matrix_counts` |
| `test_predictions.csv` → `label,prediction,<emotion>` | `y_true,y_pred,p_<emotion>`；补 run_id、dataset_id、split、actor_id |
| `selected_by_validation` | 所有候选中的选择；筛选阶段按各 comparison_group 的范围单独记录 |

现有一次 fusion 命令可能输出多个 method；统一记录时拆为各 method 的独立 run_id，引用同一原始 metrics 文件和上游来源。audio_only/visual_only 视为被引用的上游 baseline，不计作新训练重复。现有 JSON 的 `per_class` 含 aggregate 行，转成统一列表时只取八个情绪。

当前 M5 `settings` 中 hidden/heads 等参数按实际作用放入 fusion 或 classifier，不直接复制到 feature_extractors。`model_config` 字段拆分仅发生在新模板中，这次没有修改训练程序或历史结果格式，也没有实现自动转换。

## 7. 特别给 M5 的解释记录

除总分外，建议保留以下分析，能直接支持最终报告：

1. 相对 **验证集预选** 的最强 unimodal baseline 的 Macro-F1 差值；val 与 test 分开报告，不能在 test 上临时挑最强基线。
2. 哪些情绪改善或退步、主要混淆对；附同一批 sample_id 的对比案例。
3. Audio 正确/错误 × Visual 正确/错误四种情况，各有多少样本、融合后准确率如何；空组记 null。
4. decision_tuned 的 audio weight 及验证搜索记录；gated 的 gate 定义（若 z 乘 audio，则明确写出）、按情绪的分布。Gate 只代表模型内部权重，不能直接当作因果解释或模态质量结论。
5. 明确使用 learned-only 临时输入，还是 M1/M3 最终配置；两者分开比较。Cross-attention 若依赖 learned sequences，而 pooled/probs 来自别的最终配置，也应写清，不能当成完全相同输入条件。

解释性分析先按验证集设计；最终测试分析完成后用于报告，不继续据此调参。外部样本单独列数据来源、人工标注方法和样本量，不能和 RAVDESS test 混算。
