# 本地系统验证记录

2026-10-27 首轮记录：仅验证交互系统与已有模型的一致性；该轮没有训练新模型、重选test配置或增加外部情绪准确率结论。后续同日新增的外部试验见文末，二者分开记录。

- **10项回归测试通过**：`demo/.venv/bin/python -m unittest discover -s demo/tests -v`。覆盖既有val预测复现、checkpoint校验、gate、attention导出不改变预测、决策分解、固定目标敏感性、无关模态不变性、HC音频控制与独立视觉头、拒绝test回放、原始视频不按文件名取缓存。
- **11项HTTP检查通过**：`demo/.venv/bin/python -m demo.verify_http`。覆盖本地依赖状态、gate/attention/decision三种回放、原始视频WavLM＋attention、原始组合音频＋audio-only、NPZ下载及HTTP Range、跨域拒绝、路径白名单、组合不兼容拒绝、test回放拒绝、无音轨失败。多个子检查合并在同一项结果中；记录见 `test-artifacts/http_verification.json`。
- **10种可用推理方法的工程特征检查通过**：同一实际解码的合成视频特征分别进入audio-only、visual-only、concat、gated、cross-attention、bidirectional、residual+aux、LMF、equal/tuned decision。概率有限且归一化，生成相应解释。见 `test-artifacts/raw_inference_verification.json`。该记录来自首轮PCM16解码烟雾测试；最终float32解码另外完成上述HTTP重验，不混为提取精确复现证据。
- 浏览器实际操作验证了文件选择、视频上传、模型执行、已完成结果、验证回放、gate图和attention热力图；手机宽度下热力图在面板内横向滚动。
- `node --check demo/web/app.js` 及Python编译检查通过。
- 11份引入的队友源码均与 `vendor/manifest.json` 哈希一致；WavLM和M2 adapter与最终交付SHA256匹配。环境版本见 `runtime_versions.json`。

原始视频工程测试为2秒纯色画面＋正弦音轨，不含人脸或情绪真值；MediaPipe初始化和全帧检测运行后进入显式中心裁剪回退，UI显示质量警告。它验证实际解码、特征提取和推理连接，不验证真实人脸识别质量或外部语音情绪识别能力。截至首轮烟雾测试时，尚未用带真值的独立原始人脸视频完成性能评价，也未对已导出RAVDESS特征执行原视频逐样本重提取一致性验证。

当前需要队友继续补齐的是深度视觉编码器权重、完整预处理定义以及最终视觉SVM权重。原始视频使用M5视觉MLP，单独按val调权；最终论文中的视觉SVM决策融合分数不能直接用于这个UI配置。现有checkpoint和研究评价记录均保留。

## 同日更新：英文界面及独立真人视频

- HTML/JS和后端输出的界面文案均改为英文；源码对应文件无中文字符串残留。重新通过10项模型回归与11项HTTP检查，既有val预测保持一致。
- 独立于RAVDESS的24段CREMA-D标注视频全部跑通，所有384张采样帧检测到人脸。11种固定方法合计264条预测，未重新训练或重跑RAVDESS test。详见 [外部试验报告](external_evaluation_20261027/REPORT.md)。
- 浏览器实际上传一个CREMA-D的MP4副本，完成门控融合，检查英文预测、原始音频/视觉对照、64维门控及输入敏感性。截图 `ui_preview_english.png`。该副本经过浏览器兼容转码，仅作UI工程验证；评分使用官方FLV原文件，避免混淆。
- 外部样本成功推理不等于识别可靠：拼接/门控9/24，原音频10/24。不能据此宣称任意视频泛化；原RAVDESS视频逐样本提取一致性验证依旧待做。
