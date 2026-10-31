"""Render the validation-study tables, research figures and Chinese result report."""
import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import CLASSES
from .experiment import write_json


def run(study):
    study=Path(study);diag=study/'diagnostics';figs=study/'figures';figs.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'prs-fusion-study-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((study/'default_summary.json').read_text())
    rows=json.loads((study/'default_runs.json').read_text())
    interventions=json.loads((diag/'interventions.json').read_text())
    repairs=json.loads((diag/'repair_harm.json').read_text())
    groups=json.loads((diag/'complementarity_groups.json').read_text())
    attention=json.loads((diag/'attention_controls.json').read_text())
    cases=json.loads((diag/'representative_cases.json').read_text())
    status=json.loads((study/'study_status.json').read_text())
    tuning=json.loads((study/'tuning_runs.json').read_text())
    decision=json.loads((study/'tuning_decisions.json').read_text())
    scope=json.loads((diag/'scope.json').read_text())
    methods=list(summary);fusions=['concat','gated','cross_attention']
    def save(fig,name):
        fig.tight_layout();fig.savefig(figs/name,dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(np.arange(len(methods)),[summary[m]['macro_f1_mean'] for m in methods],
           yerr=[summary[m]['macro_f1_std'] for m in methods],capsize=4,color=['#8798a5','#8798a5','#4a90a4','#62a984','#c28240','#8798a5','#4a90a4'])
    for i,m in enumerate(methods):
        for row in rows:
            if row['method']==m:ax.scatter(i,row['val']['macro_f1'],color='black',s=12,zorder=3)
    ax.set(xticks=range(len(methods)),xticklabels=methods,ylabel='Validation macro-F1',ylim=(0,.8),title='Fixed actor split; seeds 42 / 43 / 44; mean ± sample SD')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');save(fig,'validation_summary.png')
    group_keys=['both_correct','audio_only_correct','visual_only_correct','both_wrong']
    fig,ax=plt.subplots(figsize=(9,4))
    for j,m in enumerate(fusions):
        vals=[np.mean([x['fusion_accuracy'] for x in groups if x['method']==m and x['group']==g and x['fusion_accuracy'] is not None]) for g in group_keys]
        ax.bar(np.arange(4)+(j-1)*.24,vals,width=.24,label=m)
    ax.set(xticks=range(4),xticklabels=group_keys,ylabel='Fusion accuracy within group',ylim=(0,1.05),title='Validation error complementarity; mean over seeds')
    ax.legend();plt.setp(ax.get_xticklabels(),rotation=15,ha='right');save(fig,'error_complementarity.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,modality in zip(axes,['audio','visual']):
        for m in fusions:
            points=[summary[m]['macro_f1_mean']]+[np.mean([x['macro_f1'] for x in interventions if x['method']==m and x['condition']==f'{modality}_noise_{level}']) for level in ['0.25','0.5','1.0']]
            ax.plot([0,.25,.5,1],points,marker='o',label=m)
        ax.set(xlabel='Noise SD in standardized feature units',ylabel='Validation macro-F1',title=modality+' feature noise (not raw signal noise)');ax.legend();ax.grid(alpha=.2)
    save(fig,'feature_noise_curves.png')
    gate=[]
    with (diag/'gates_val.csv').open() as f:gate=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(9,4))
    ax.boxplot([[float(x['mean_audio_gate']) for x in gate if int(x['label'])==c] for c in range(8)],tick_labels=CLASSES)
    ax.set(ylabel='Mean gate across hidden dimensions',title='Descriptive gate distributions; not modality contribution percentages')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');save(fig,'gate_by_emotion.png')
    for category in ('correct','incorrect'):
        with np.load(diag/f'attention_{category}_s42.npz',allow_pickle=False) as z:
            weights=z['weights'];sid=str(z['sample_id'])
        fig,axes=plt.subplots(1,len(weights),figsize=(14,4),sharey=True)
        for h,ax in enumerate(axes):
            ax.imshow(weights[h],aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=float(weights.max()));ax.set(xlabel='Visual token',title=f'Head {h+1}')
        axes[0].set_ylabel('Audio token (not seconds)');fig.suptitle(f'Validation {category}: {sid}',y=1.02)
        save(fig,f'attention_{category}_heads.png')
    # Plain tables preserve seed-level observations alongside aggregates.
    with (study/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','n','seeds','val_macro_f1_mean','val_macro_f1_sample_std','val_accuracy_mean','val_accuracy_sample_std','test_status'])
        for m,s in summary.items():w.writerow([m,s['n'],'42/43/44',s['macro_f1_mean'],s['macro_f1_std'],s['accuracy_mean'],s['accuracy_std'],'not_run'])
    def avg(items,key):return float(np.mean([r[key] for r in items]))
    lines=['# 多模态融合验证实验与解释分析','','日期：2026-10-19。仅固定正式演员划分的train=960、val=240，test保留。所有分数是同一验证集上的开发结果，不是最终泛化成绩。决策融合仍等待上游概率类别顺序确认。','','## 本轮执行范围','',
           f"完成7种方法×3个seed共21组默认配置，其中复用首轮3组，经预测核验一致；另做4组定向调参，本轮新增训练{status['new_fits']}次，合计保存25组配置/seed结果。",'',
           '- Seed统一为42/43/44，保持已采用的团队记录规范；不使用早期草案的123/2026，不改变actor split。',
           '- 原始测试特征和上游概率没有进入本轮：只加载上轮1200条train/val缓存，所有checkpoint/预测/normalizer只针对开发集。',
           '- 拼接/门控输入176D音频和576D视觉；attention输入128D learned音频序列和16×576视觉序列。',
           '- 单模态控制：相同pooled特征、隐藏层64的MLP；不是队友原始分类器成绩。序列控制：音频序列均值→MLP、两路序列均值拼接→MLP。',
           '- 训练保持AdamW lr0.001、weight_decay1e-4、batch64、dropout0.2、最多30epochs、patience8；仅用验证Macro-F1选epoch。',
           '- Visual的16个token暂按有效处理；上游checkpoint、训练来源和最终视觉配置仍待确认。','','## 多 seed 验证结果','','| 方法 | 验证 Macro-F1 均值±样本标准差 | 验证 Accuracy 均值 |','|---|---:|---:|']
    for m,s in summary.items():lines.append(f"| {m} | {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f} | {s['accuracy_mean']:.4f} |")
    lines+=['','标准差使用ddof=1、n=3；三个seed仅表示固定演员划分下的训练波动，不能据此声称统计显著或跨演员划分稳定。所有默认单seed结果保存在default_runs.json。',
            '',f"验证均值最高为cross_attention，比同协议visual_only控制高{100*(summary['cross_attention']['macro_f1_mean']-summary['visual_only']['macro_f1_mean']):.2f}个百分点，比concat高{100*(summary['cross_attention']['macro_f1_mean']-summary['concat']['macro_f1_mean']):.2f}个百分点。Cross-attention并非每个seed都胜过concat；不得称为稳定压倒性优势。",'',
            f"相比两路序列均值拼接控制，cross_attention高{100*(summary['cross_attention']['macro_f1_mean']-summary['sequence_mean_concat']['macro_f1_mean']):.2f}个百分点。该差异仍包含投影、残差、LayerNorm、参数量及attention等结构差异，不能单独归因于时间对齐。",'',
            '当前结果支持这些固定表示在验证集上存在互补收益，但尚未与已选定的最终上游单模态流水线进行完整对照，不能宣称优于团队所有单模态方案。',
            '', '![验证均值与波动](figures/validation_summary.png)','','## 定向改进实验','','按三seed默认均值选择cross_attention、concat，仅在seed42分别试hidden32或dropout0.4，不同时改变两个因素。预设追加seed门槛为seed42提升至少0.02。此门槛用于控制工作量，不是显著性标准。','','| 方法 | hidden | dropout | seed42 验证 Macro-F1 |','|---|---:|---:|---:|']
    for t in tuning:lines.append(f"| {t['method']} | {t['hidden']} | {t['dropout']} | {t['val']['macro_f1']:.4f} |")
    lines+=['','Cross-attention增加dropout后seed42得到0.6828，但只比默认提升约1.55个百分点，未达到追加门槛。其他候选也未达到；因此没有进一步扩大搜索，不把这个单seed调参最高值作为正式主结果。',
            '', '## 错误互补与修复破坏','','以三seed均值选出的visual_only为共同基线，各seed对应自己的分类器。以下是每个seed的样本数再取平均，不是整数样本合并后的独立检验。','','| 方法 | 每seed平均修复数 | 每seed平均破坏数 |','|---|---:|---:|']
    for m in fusions:
        rr=[x for x in repairs if x['method']==m];lines.append(f"| {m} | {avg(rr,'repaired'):.2f} | {avg(rr,'harmed'):.2f} |")
    lines+=['','Cross-attention平均修复48条、破坏18.33条；concat修复更多但也破坏更多。收益应同时看修复和破坏，计数净变化不直接等于Macro-F1变化。',
            '', '![错误互补分组](figures/error_complementarity.png)','','## 模态干预','','固定checkpoint，不重新训练。将标准化后的音频/视觉替换为零，或将视觉替换成不同情绪样本；本轮错误配对在各固定推理batch内选择。下表为三seed平均Macro-F1。','','| 方法 | 正常 | 音频置零 | 视觉置零 | 视觉错误配对 |','|---|---:|---:|---:|---:|']
    for m in fusions:
        vals=[avg([x for x in interventions if x['method']==m and x['condition']==c],'macro_f1') for c in ('audio_zero','visual_zero','visual_mismatched')]
        lines.append(f"| {m} | {summary[m]['macro_f1_mean']:.4f} | "+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','两路置零均损害预测，视觉错误配对对attention影响尤其大，支持当前模型对正确配对的两路表示敏感。不能把下降量换算成模态贡献百分比或因果效应。',
            '', '![特征噪声曲线](figures/feature_noise_curves.png)',
            '', '噪声是在标准化特征上添加独立高斯扰动，不是原始音频SNR或视频亮度退化。独立序列噪声可以被时间平均削弱，因此attention对音频噪声相对不敏感，不等于忽略音频或真实音频稳健。',
            '', '## 门控解释','','已导出全部验证样本的平均gate、情绪分布，以及音频/视觉噪声前后的变化。音频噪声下平均gate仅约+0.0008～+0.0012；视觉噪声下变化约-0.0011～+0.0002，没有清晰的可靠模态切换证据。Gate均值不等于音频贡献比例；不能写成“模型已经学会自适应降低噪声模态权重”。',
            '', '![门控分布](figures/gate_by_emotion.png)','','## Attention是否提供可信解释','','保留Q/V投影、audio residual、LayerNorm和分类头，只把注意力权重替换成有效帧上的均匀权重。额外用高权重/低权重/随机4个视觉token置零对照，随机对照重复5次。原始attention权重重构logits已与正常forward一致。','','| 干预 | 平均验证 Macro-F1 | 原预测类别概率平均下降 |','|---|---:|---:|']
    for c in ('uniform_weights','high_frames','low_frames','random_frames'):
        rr=[x for x in attention if x['condition']==c];lines.append(f"| {c} | {avg(rr,'macro_f1'):.4f} | {avg(rr,'original_predicted_class_probability_drop'):.4f} |")
    lines+=['','**均匀权重得到0.6627，与原模型0.6608相近。当前没有证据证明learned attention权重的精细选择是性能优势的主要来源。** 高权重token扰动对原预测类别概率的影响大于随机对照，但Macro-F1并未呈现同样的严格排序；这是局部敏感性证据，不能把热图作为因果解释。额外投影、残差及分类头可能同样重要，后续需要更严格的结构消融。',
            '', '以下按seed42、sample ID顺序选第一条正确/错误案例，展示全部4个head；不是按热图美观程度选样本。纵轴是音频token，不是秒；缺原始时间戳时不能声称对应某个发音或表情时刻。',
            '', '![正确案例](figures/attention_correct_heads.png)','','![错误案例](figures/attention_incorrect_heads.png)',
            '', '## 代表性成功与失败案例','','固定规则：seed42、三seed均值最优融合方法、每类按sample ID选前两条；所有分组及完整概率见diagnostics/representative_cases.json。','','| 情况 | sample_id | 标签 | 音频控制 | 视觉控制 | 融合 |','|---|---|---|---|---|---|']
    for c in cases:lines.append('| '+c['category']+' | '+c['sample_id']+' | '+CLASSES[c['label']]+' | '+CLASSES[c['audio_prediction']]+' | '+CLASSES[c['visual_prediction']]+' | '+CLASSES[c['fusion_prediction']]+' |')
    lines+=['','## 验证与记录','','- 共14项自动测试通过，覆盖开发集接口隔离、padding、单模态控制不依赖另一模态、attention权重重构与均匀权重的mask行为。','- 新训练checkpoint均重新加载后复现验证预测，复用的3个checkpoint重新验证混淆矩阵一致。','- 全部预测文件为val，240条；开发manifest为train/val共1200条。没有真实test指标、预测或test checkpoint选择。','- 每run保留run.json、配置、checkpoint、逐epoch历史及验证概率；default_runs/default_summary、tuning_runs/decisions、diagnostics分别保存主结果、探索和解释。','- 各方法当前使用同一上游特征与actor split；上游run_id/训练来源待补，视觉16token均有效仍是明示假设。',
            '', '## 下一步与暂缓项','','1. 当前可继续使用concat作为简单参照、cross-attention作为候选；不要因单seed0.6828而更换主配置。','2. 与上游确认概率类别顺序、完整训练来源、16帧有效性及M3最终视觉A/B/A+B选择后，再补决策融合和最终流水线比较。','3. 如需证明动态attention有效，进一步做参数量匹配的投影/残差对照及均匀attention重训，而非仅展示热图。','4. 原始音视频退化、外部视频演示、最终测试及不同演员划分扩展本轮未运行。测试集依照此前约定继续保留。','5. 验证集已用于选择epoch和探索配置，结果有选择偏差；不能把重复seed当作新的独立数据或最终泛化检验。',
            '', '## 复跑命令','','从multimodal_fusion目录使用新输出路径：','','```bash','python -m fusion.study --pilot results/m5_feature_pilot_20261019_s42 --output results/new_validation_study',
            'python -m fusion.study_diagnostics --study results/new_validation_study --pilot results/m5_feature_pilot_20261019_s42','python -m fusion.study_report --study results/new_validation_study','```',
            '', 'study只接受已适配train/val缓存并校验范围，不读取上游概率或test；输出目录已存在则拒绝覆盖。']
    (study/'实验结果与解释分析.md').write_text('\n'.join(lines)+'\n')
    print('Report and figures saved:',study/'实验结果与解释分析.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);run(p.parse_args().study)
