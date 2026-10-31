"""Independently validate structure-study artifacts and summarize paired contrasts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from . import CLASSES
from .experiment import write_json
from .feature_inputs import sha256


def validate(out):
    out=Path(out);plan=json.loads((out/'plan.json').read_text());rows=json.loads((out/'runs.json').read_text())
    expected={(m,s) for m in plan['methods'] for s in plan['seeds']}
    assert {(r['method'],r['seed']) for r in rows}==expected and len(rows)==len(expected)
    assert plan['test_status']=='not_run' and not plan['uses_wavlm']
    with (out/'input_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    lookup={r['sample_id']:r for r in manifest}
    assert len(lookup)==len(manifest)==1200
    assert sum(r['split']=='val' for r in manifest)==240
    assert sum(r['split']=='train' for r in manifest)==960
    assert not ({r['actor_id'] for r in manifest if r['split']=='train'} & {r['actor_id'] for r in manifest if r['split']=='val'})
    template=json.loads((Path(__file__).resolve().parents[2]/'docs/templates/run.template.json').read_text())
    def keys(r,t):
        if isinstance(t,dict):
            assert set(t)<=set(r)
            for k,v in t.items():keys(r[k],v)
        elif isinstance(t,list) and isinstance(r,list) and t:
            for item in r:keys(item,t[0])
    for row in rows:
        folder=Path(row['folder']);r=json.loads((folder/'run.json').read_text());keys(r,template)
        assert r['test_status']=='not_run' and r['metrics_by_split']['test'] is None
        assert r['artifacts']['predictions_test_ref'] is None
        assert sha256(folder/'checkpoint.pt')==r['artifacts']['checkpoint_sha256']
        assert r['training']['other_parameters']['plan_sha256']==sha256(out/'plan.json')
        assert sha256(out/'input_manifest.csv')==r['input_manifest_sha256']
        assert not list(folder.glob('*test*'))
        with (folder/'predictions_val.csv').open() as f:p=list(csv.DictReader(f))
        assert len(p)==len({x['sample_id'] for x in p})==240
        for x in p:
            meta=lookup[x['sample_id']]
            assert x['split']==meta['split']=='val' and x['actor_id']==meta['actor_id']
            assert int(x['y_true'])==int(meta['label'])==int(x['sample_id'].split('-')[1])-1
            assert x['run_id']==r['run_id']
        y=np.array([int(x['y_true']) for x in p]);pred=np.array([int(x['y_pred']) for x in p])
        prob=np.array([[float(x['p_'+c]) for c in CLASSES] for x in p])
        assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all()
        assert np.allclose(prob.sum(1),1,atol=1e-5) and np.array_equal(prob.argmax(1),pred)
        precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
        m=r['metrics_by_split']['val'];assert support.tolist()==[16]+[32]*7
        for name,value in [('accuracy',accuracy_score(y,pred)),('macro_f1',f1.mean()),('macro_precision',precision.mean()),('macro_recall',recall.mean())]:
            assert abs(value-m[name])<1e-12 and abs(value-row['val'][name])<1e-12
        assert m['confusion_matrix_counts']==confusion_matrix(y,pred,labels=range(8)).tolist()
        for i,c in enumerate(m['per_class']):
            assert c['class_name']==CLASSES[i] and c['support']==int(support[i])
            assert abs(c['precision']-precision[i])<1e-12 and abs(c['recall']-recall[i])<1e-12 and abs(c['f1']-f1[i])<1e-12
        h=json.loads((folder/'history.json').read_text())
        winner=max(h,key=lambda x:x['val_macro_f1'])
        assert winner['epoch']==row['best_epoch'] and abs(winner['val_macro_f1']-m['macro_f1'])<1e-12
        assert row['checkpoint_roundtrip_verified']
    for name,digest in json.loads((out/'code_hashes.json').read_text()).items():assert sha256(out/'code'/name)==digest
    for b in plan['base_references'].values():assert sha256(b['checkpoint_ref'])==b['checkpoint_sha256']
    result=dict(records=len(rows),prediction_files=len(rows),prediction_rows=len(rows)*240,
                note='Same 240 validation recordings repeated across models/seeds, not independent samples.',
                metrics_schema_split_id_history_hash_checks='passed',test_status='not_run',uses_wavlm=False)
    write_json(out/'validation_report.json',result)
    return rows


def report(out):
    out=Path(out);rows=validate(out);baseline=json.loads((out/'reused_baselines.json').read_text())
    combined=baseline+rows
    def paired(candidate,control):
        a={r['seed']:r['val']['macro_f1'] for r in combined if r['method']==candidate}
        b={r['seed']:r['val']['macro_f1'] for r in combined if r['method']==control}
        values=[a[s]-b[s] for s in sorted(a)]
        return dict(candidate=candidate,control=control,differences=values,mean_difference=float(np.mean(values)),
                    positive_seeds=sum(v>0 for v in values),followup_gate=bool(np.mean(values)>=.02 and sum(v>0 for v in values)>=2))
    pairs=[('audio_residual','concat'),('audio_residual_aux','audio_residual'),
           ('audio_residual_mdrop','audio_residual'),('audio_residual_mdrop_aux','audio_residual_aux'),
           ('audio_residual_aux','concat'),('audio_residual_mdrop_aux','concat')]
    pairs += [(m,'cross_attention') for m in ('projected_mean','uniform_attention','cross_position','cross_temporal','cross_bidirectional')]
    contrasts=[paired(*p) for p in pairs];write_json(out/'paired_contrasts.json',contrasts)
    methods=list(dict.fromkeys(r['method'] for r in combined))
    summaries={}
    for m in methods:
        group=[r for r in combined if r['method']==m]
        values=[r['val']['macro_f1'] for r in group]
        summaries[m]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),
                          accuracy=float(np.mean([r['val']['accuracy'] for r in group])))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7))
    colors=['#85929e' if m in {r['method'] for r in baseline} else '#2374ab' for m in methods]
    ax.barh(range(len(methods)),[summaries[m]['mean'] for m in methods],
            xerr=[summaries[m]['sd'] for m in methods],color=colors,capsize=3)
    ax.set_yticks(range(len(methods)),methods);ax.invert_yaxis();ax.set_xlim(0,.85)
    ax.set_xlabel('Validation Macro-F1 (mean and sample SD across seeds 42/43/44)')
    ax.set_title('Historical features: residual and sequence structure controls\nValidation only; test not evaluated in this study')
    for i,m in enumerate(methods):ax.text(summaries[m]['mean']+summaries[m]['sd']+.008,i,f'{summaries[m]["mean"]:.3f}',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    by_pair={(r['candidate'],r['control']):r for r in contrasts}
    aux=by_pair[('audio_residual_aux','concat')]
    bidir=by_pair[('cross_bidirectional','cross_attention')]
    uniform=by_pair[('uniform_attention','cross_attention')]
    lines=['# 本轮结构实验解读\n','未接入WavLM，27次新训练，仅评价旧特征的240条验证样本；原测试已暴露，本轮为探索。\n',
           f'残差+辅助CE相对concat平均F1变化{aux["mean_difference"]:+.4f}，{aux["positive_seeds"]}/3个seed改善；是否达到预设追加投入门槛：{aux["followup_gate"]}。',
           f'Uniform重训相对原attention变化{uniform["mean_difference"]:+.4f}，{uniform["positive_seeds"]}/3个seed改善；双向attention变化{bidir["mean_difference"]:+.4f}，{bidir["positive_seeds"]}/3个seed改善。微小均值差不等于稳定优势。',
           '本轮可保留辅助CE作为最终输入定稿后的复跑候选；无需根据单个seed继续扩大复杂attention搜索。\n',
           '![验证结果](validation_comparison.png)\n','## 主要对照\n',
           '| 新方法 | 对照 | 验证F1均值差 | 三seed差值 | 正向seed |',
           '|---|---|---:|---|---:|']
    for r in contrasts:lines.append(f'| {r["candidate"]} | {r["control"]} | {r["mean_difference"]:+.4f} | '+', '.join(f'{d:+.4f}' for d in r['differences'])+f' | {r["positive_seeds"]}/3 |')
    lines+=['\n实用投入门槛为平均提升0.02且至少2/3个seed改善；不是显著性检验。残差超过音频单模态并不足以说明胜过简单融合，必须同时看与concat/gated的比较。',
            '残差4组使用同一结构和初始化做dropout×辅助CE的2×2对照；其基础音频头固定，额外训练预算为至多30epoch。不能把训练总成本视为与从零训练concat完全相同。',
            '序列组与旧attention使用相同RandomState(seed)批次顺序和初始化来源；uniform取消了注意力选择和注意力权重dropout，Q/K参数不参与学习。',
            '位置编码是相对片段索引；不同模态裁剪/采样起点未证实一致。双向方法增加参数量。',
            '新音频交付后，先按其pooling语义生成train/val缓存，并用相同特征训练seed对应audio-only，再运行structure_study；更新source_run元数据，不混用本轮旧音频头。']
    lines+=['\n## 验证集修复与破坏（相对于固定音频头）\n','| 方法 | 修复音频错误数/seed | 破坏音频正确数/seed |','|---|---:|---:|']
    for m in [x for x in methods if x.startswith('audio_residual')]:
        g=[r for r in rows if r['method']==m]
        lines.append(f'| {m} | {np.mean([r["repairs_audio"] for r in g]):.2f} | {np.mean([r["harms_audio"] for r in g]):.2f} |')
    lines+=['\n所有方法及accuracy见[完整结果](实验结果.md)；单run保留逐演员、置零模态、修复/破坏诊断。没有测试推理，不把本轮验证分数当未来测试保证。']
    (out/'结果解读.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'report_code_hash.json',{'structure_report.py':sha256(__file__)})
    print(json.dumps({'validation':'passed','contrasts':contrasts},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
    report(p.parse_args().study)
