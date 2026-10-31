# M5 本地视频融合演示

上传带音轨的短视频，在本机解码、提取 WavLM 音频及 DWT 人脸特征，选择已训练的融合方法，查看预测分布和对应的解释。支持显式的验证特征回放，方便比较已交付的其他表示。两种模式不会通过文件名相互替代。

## 启动

在项目根目录执行：

```bash
demo/.venv/bin/python -m demo.server --port 8765
```

浏览器打开 <http://127.0.0.1:8765>。服务只监听本机地址。首次推理需要加载 WavLM；之后模型留在内存。输入限制为带音轨、30秒以内、100MB以内的视频，适合单人、正面、短语音片段。多脸时遵循 M3 原实现，逐帧选择最高检测分数的人脸，不提供多人物身份跟踪。

## 当前可用范围

| 项目 | 原始视频上传 | 已有验证特征回放 |
|---|---|---|
| WavLM learned128 | 已接通，固定 M2 原版本 | 已接通 |
| Audio HC48 / combined176 | 已接通 M1 原特征代码；仅匹配 audio-only 控制 | 已接通 audio-only 控制；没有相应多模态训练头的组合不开放 |
| Visual HC320 | 已接通，MediaPipe对齐＋Haar DWT | 已接通 |
| Visual deep576 / combined896 | 等待队友原编码器及预处理权重 | 已接通 |
| 拼接 / 门控 / 单向attention | 已接通 | 根据匹配checkpoint开放 |
| 双向attention / residual+aux / LMF | 已接通 HC 输入 | 根据匹配checkpoint开放 |
| 等权 / 验证调权概率融合 | 已接通，使用下面单独说明的视觉MLP协议 | 使用交付概率及原权重 |

所有交互神经模型固定使用 seed42；这只是预先固定的演示实例，不能把界面分数当作报告中三seed均值，也没有按test重新挑选seed。特征维度和checkpoint必须匹配，不能把任意提取器插入原融合头。

## 原始视频提取与来源

- 音频：FFmpeg解码为float32 WAV并保留原采样率/声道，避免额外PCM16量化 → 由Librosa按原M1/M2方式转mono16k并trim35dB → 固定 `microsoft/wavlm-base-plus` revision `4c66d4806a428f2e922ccfa1a962776e232d487b` → layer6 → 与M2缓存相同的float16再float32 → 原 temporal adapter/context/attentive-statistics pooling/head。pooled128不是sequence均值。视频音轨与官方audio-only WAV的编码本身仍可能不同，未声称原视频重提取逐样本精确复现既有导出。
- 可选手工音频调用固定M1的 `extract_recording_features`，依照实际交付CSV的48个特征列重排并校验；组合音频为HC48在前、WavLM128在后。它们只开放匹配的audio-only模型，不能套用原WavLM多模态融合头。内部保留WavLM序列以维持已训练控制模型的数据接口，但不会把它替代所选pooled输入。
- WavLM权重SHA256：`3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`。
- M2 adapter SHA256：`0db8f833f3d32f21b676c7ed008acdf094bb20dda31e3498e642249d597f04bb`。
- 视觉：均匀16帧 → MediaPipe model0/confidence0.5 → 眼部旋转对齐/1.3框margin/224RGB → 内裁180/Haar/4×4网格/每子带五统计 → 16×320 → 均值320。调用 `vendor/` 中固定的队友函数，不重新猜测公式。
- 固定源码来自本地已fetch的团队提交 `556e56595a32f469fa99ce469234aa2c8d42a297`；`vendor/manifest.json` 记录逐文件哈希。引入的源码未改动。
- 未检出脸时按M3流程中心裁剪并显示警告；缺失帧借邻近框。用户也可显式选择中心裁剪作为预处理诊断。该分布变化不具备已验证的情绪识别质量。
- 每次上传以文件内容SHA256生成external ID，不从文件名推断标签，不按文件名查找缓存。音频trim起点和估计CNN token中心、16帧时间均保留。它们帮助展示输入位置，不构成已证明的情绪时间对齐。

## 决策融合需要特别区分

当前没有队友最终视觉SVM的 `.joblib`，因此原始视频的视觉概率使用**现有 M5 visual-only MLP seed42**。音频仍使用原M2头。界面明确显示此区别。

针对这组概率，`Engine.raw_visual_alpha()`只读取已有240条val预测，按Macro-F1在0到1步长0.05选择权重，得到音频 **α=0.70**；不读取test，不重训分类器。详细grid随原始视频调权决策结果JSON导出，协议名称为 `demo_audio_original_visual_M5_MLP_seed42_validation_tuned`。它与论文中使用交付视觉SVM概率、α=0.60的HC决策融合**不是同一配置**，不能套用论文的0.8349测试F1。

验证特征回放仍使用交付原概率，HC/deep/combined的α分别0.60/0.70/0.70。不要把原始视频的替代视觉头与回放原概率混为一项研究结果。队友交付视觉头后，应验证其class order、预处理、val逐样本概率，再新增对应的决策协议。

## 各方法的解释

| 方法 | 界面解释 | 含义边界 |
|---|---|---|
| Decision | 每类α·P_audio与(1−α)·P_visual的堆叠分解 | 精确算法分解，不是因果贡献 |
| Concat / LMF / 单模态控制 | 固定原预测类别，分别替换整路标准化特征为0后的概率差 | 0代表训练均值；可能分布外，不是现实音视频SNR或质量评分 |
| Gated | 64维gate、均值、固定gate0.5后的预测 | gate偏向投影分支，不等于模态贡献百分比或可靠性 |
| Cross-attention | 四head平均、只取有效token并最多48个音频分桶的音频→视觉热力图 | 显示内部关联；导出权重时核对logits不变，不宣称因果或同步成功 |
| Bidirectional | 同上，并导出反向attention摘要 | 增加参数的结构，不能把改进单归因为方向 |
| Frozen-audio residual | 原音频logits与修正支路的逐类有符号差值 | 修正同时读取两种模态，不是纯视觉因果贡献 |

单片段无ground truth时仅显示预测概率。概率未经校准，最大概率不是准确率，也不是人类真实情绪的可信度。

## 文件与复现

- `server.py`：仅loopback HTTP服务、上传限制、串行推理队列、job状态及下载。
- `engine.py`：读取最终checkpoint，核对锁定哈希，提供兼容选项、回放及解释。
- `extraction.py`：实际音视频提取和时序/质量记录。
- `web/`：无CDN依赖的HTML/CSS/JS界面。
- `sessions/<uuid>/`：上传原文件、`audio.wav`、16张人脸、`extracted_features.npz`、`result.json`。重启后内存job索引不恢复，文件仍保留；上传内容不会发送到远程推理服务。
- `assets/`：约370MB固定上游权重，已加入gitignore。交互推理只用本地文件，不会临时自动下载别的模型。

如需在别的机器配置：

```bash
# 在项目根目录；先安装M5基础依赖，或复用已有mr环境
/path/to/mr/bin/python -m venv --system-site-packages demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python demo/setup_assets.py
demo/.venv/bin/python -m demo.server
```

还需要本项目最终实验的checkpoint和 `evaluation_lock.json`、验证预测CSV、development.npz及M2 adapter。它们当前已在本机，未重新训练，也不由浏览器上传。仅复制demo源码不足以复制这些研究模型。Mac上的MediaPipe需要正常图形服务权限；不要在禁止图形服务的沙箱里把初始化失败悄悄当作“没检测到脸”。

## 验证

```bash
demo/.venv/bin/python -m unittest discover -s demo/tests -v
node --check demo/web/app.js
```

回归检查包括：UI推理复现记录的val概率、锁定checkpoint哈希、gate边界、attention导出预测不变、加权项相加一致、单模态无关支路干预不变、拒绝test回放与不兼容组合、原始视频不按文件名走缓存。

工程测试使用人工生成的纯色视频＋正弦音轨，验证真实上传、解码、提取、融合及结果导出；它没有情绪标签，**不属于新实验准确率，也不是外部泛化验证**。已有val特征回放仍属于曾参与模型选择的数据。全组报告现在可以说明“已有本机推理UI”。2026-10-27 已补充独立 CREMA-D 的小规模外部评价，见下方；工程烟雾测试和外部情绪评价分开记录。

## 英文 UI 与外部视频初测（2026-10-27）

界面已改为英文，包括设置、处理进度、错误、解释图、下载和概率说明。打开 http://127.0.0.1:8765/ ，若已有旧页面请刷新。浏览器原生播放器控件的语言可能跟随系统语言。

实际使用 CREMA-D 官方标注的 24 段视频（4位演员 × 6类情绪，同一句台词），固定现有 seed42 模型直接推理，没有依据外部分数调参。24/24 处理成功，384/384采样帧检测到人脸。原音频头准确率41.7%，拼接/门控37.5%，单向cross-attention20.8%；这说明系统能够处理外部视频，但尚不能认定具有可靠泛化能力。小样本、领域差异和待完成的原视频提取一致性核查都需要保留在结论中。

完整协议、所有11种方法、逐样本预测、限制和后续安排：[外部视频评价报告](external_evaluation_20261027/REPORT.md)。单独保留团队v2格式的11份实验记录，不与RAVDESS test混算，不覆盖原有结果。复现入口为 `external_eval.py` 与 `external_report.py`；公开数据视频仅保存在gitignore目录。
