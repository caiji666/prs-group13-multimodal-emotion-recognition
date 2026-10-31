"""Recompute all candidate/diagnostic metrics and report the bounded literature study."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,precision_recall_fscore_support,f1_score
from . import CLASSES
from .experiment import write_json
from .feature_inputs import sha256


def check_metrics(y,pred,m):
    pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
    assert len(y)==m['n']==240 and support.tolist()==[16]+[32]*7
    for k,v in [('accuracy',accuracy_score(y,pred)),('macro_f1',f1.mean()),('macro_precision',pr.mean()),('macro_recall',re.mean())]:assert abs(m[k]-v)<1e-12
    assert confusion_matrix(y,pred,labels=range(8)).tolist()==m['confusion_matrix_counts']
    for i,c in enumerate(m['per_class']):
        assert c['class_name']==CLASSES[i] and c['support']==int(support[i])
        for k,v in [('precision',pr[i]),('recall',re[i]),('f1',f1[i])]:assert abs(c[k]-v)<1e-12


def validate(out):
    rows=json.loads((out/'runs.json').read_text());plan=json.loads((out/'plan.json').read_text())
    assert len(rows)==42 and sum(r['family']=='neural' for r in rows)==15
    assert not plan['uses_wavlm'] and plan['test_status']=='not_run'
    neural=[r for r in rows if r['family']=='neural'];svm=[r for r in rows if r['family']!='neural']
    assert {(r['method'],r['seed']) for r in neural}=={(m,s) for m in plan['methods'] for s in plan['seeds']}
    assert len(svm)==27 and all(r['seed'] is None for r in svm)
    config_keys={(r['configuration']['modality'],r['configuration']['kernel'],r['configuration']['C'],r['configuration']['alpha']) for r in svm}
    expected={(m,k,c,None) for m in ('audio','visual','concat') for k in ('linear','rbf') for c in (.1,1.,10.)}
    expected|={('kernel_fusion','precomputed',c,a) for a in (.25,.5,.75) for c in (.1,1.,10.)}
    assert config_keys==expected
    with (out/'input_manifest.csv').open() as f:manifest=list(csv.DictReader(f))
    lookup={r['sample_id']:r for r in manifest};assert len(lookup)==len(manifest)==1200
    assert {r['split'] for r in manifest}=={'train','val'}
    assert not({r['actor_id'] for r in manifest if r['split']=='train'}&{r['actor_id'] for r in manifest if r['split']=='val'})
    template=json.loads((Path(__file__).resolve().parents[2]/'docs/templates/run.template.json').read_text())
    def keys(r,t):
        if isinstance(t,dict):
            assert set(t)<=set(r)
            for k,v in t.items():keys(r[k],v)
        elif isinstance(t,list) and isinstance(r,list) and t:
            for x in r:keys(x,t[0])
    for row in rows:
        folder=Path(row['folder']);r=json.loads((folder/'run.json').read_text());keys(r,template)
        assert r['test_status']==row['test_status']=='not_run' and r['metrics_by_split']['test'] is None
        assert r['artifacts']['predictions_test_ref'] is None and not list(folder.glob('*test*'))
        assert sha256(r['artifacts']['checkpoint_ref'])==r['artifacts']['checkpoint_sha256']
        assert sha256(out/'plan.json')==r['training']['other_parameters']['plan_sha256']
        assert sha256(out/'input_manifest.csv')==r['input_manifest_sha256']
        with (folder/'predictions_val.csv').open() as f:p=list(csv.DictReader(f))
        assert len(p)==len({x['sample_id'] for x in p})==240
        for x in p:
            meta=lookup[x['sample_id']]
            assert x['split']==meta['split']=='val' and x['actor_id']==meta['actor_id']
            assert int(x['y_true'])==int(meta['label'])==int(x['sample_id'].split('-')[1])-1
            assert x['run_id']==r['run_id']
        y=np.array([int(x['y_true']) for x in p]);pred=np.array([int(x['y_pred']) for x in p])
        check_metrics(y,pred,row['val']);check_metrics(y,pred,r['metrics_by_split']['val'])
        if row['family']=='neural':
            q=np.array([[float(x['p_'+c]) for c in CLASSES] for x in p]);assert np.isfinite(q).all()
            assert (q>=0).all() and (q<=1).all() and np.allclose(q.sum(1),1,atol=1e-5) and np.array_equal(q.argmax(1),pred)
            h=json.loads((folder/'history.json').read_text());winner=max(h,key=lambda v:v['val_macro_f1'])
            assert winner['epoch']==row['best_epoch'] and abs(winner['val_macro_f1']-row['val']['macro_f1'])<1e-12
            if row['method']=='aux_control':assert abs(row['historical_aux_f1_difference'])<1e-12
        else:
            assert all(x['p_'+c]=='' for x in p for c in CLASSES)
            assert r['pipeline']['classifier']['probability_method']=='none'
            with (folder/'decision_scores_val.csv').open() as f:s=list(csv.DictReader(f))
            assert [x['sample_id'] for x in s]==[x['sample_id'] for x in p]
            scores=np.array([[float(x['score_'+c]) for c in CLASSES] for x in s])
            assert np.isfinite(scores).all() and np.array_equal(scores.argmax(1),pred)
        assert row['checkpoint_roundtrip_verified']
    selection=json.loads((out/'svm_selection.json').read_text())
    for family,rid in selection['winners'].items():
        candidates=[r for r in svm if r['family']==family];winner=max(candidates,key=lambda r:r['val']['macro_f1'])
        assert winner['run_id']==rid and winner['selected_in_family']
    gates=json.loads((out/'gate_diagnostic/results.json').read_text());assert len(gates)==138
    for seed in (42,43,44):
        with np.load(out/f'gate_diagnostic/gate_predictions_s{seed}.npz',allow_pickle=False) as z:
            gp=z['probabilities'];gy=z['labels'];conditions=z['conditions'];repeats=z['repeats'];ids=z['sample_ids']
        assert len(gp)==46 and len(ids)==240 and all(lookup[s]['split']=='val' for s in ids)
        assert np.array_equal(gy,np.array([int(lookup[s]['label']) for s in ids]))
        with np.load(out/f'gate_diagnostic/gate_values_s{seed}.npz',allow_pickle=False) as z:
            assert z['gates'].shape==(240,64) and np.isfinite(z['gates']).all()
            assert ((z['gates']>=0)&(z['gates']<=1)).all()
            assert z['permutations'].shape==(20,240)
            for permutation in z['permutations']:assert np.array_equal(np.sort(permutation),np.arange(240))
        assert np.isfinite(gp).all() and np.allclose(gp.sum(-1),1,atol=1e-5)
        original=gp[0];target=original.argmax(1)
        for q,condition,repeat in zip(gp,conditions,repeats):
            r=next(r for r in gates if r['seed']==seed and r['condition']==condition and r['repeat']==(None if repeat<0 else int(repeat)))
            assert abs(r['macro_f1']-f1_score(gy,q.argmax(1),labels=range(8),average='macro',zero_division=0))<1e-12
            assert abs(r['accuracy']-(gy==q.argmax(1)).mean())<1e-12
            assert abs(r['prediction_change_rate']-(target!=q.argmax(1)).mean())<1e-12
            assert abs(r['mean_absolute_probability_change']-np.abs(original-q).mean())<1e-12
            assert abs(r['original_predicted_probability_drop']-(original[np.arange(240),target]-q[np.arange(240),target]).mean())<1e-12
    for k,digest in json.loads((out/'code_hashes.json').read_text()).items():assert sha256(out/'code'/k)==digest
    for r in plan['baseline_refs'].values():assert sha256(r['checkpoint_ref'])==r['checkpoint_sha256']
    result=dict(records=42,neural_records=15,deterministic_svm_candidates=27,prediction_files=42,
                gate_conditions_verified=138,aux_controls_reproduced=3,all_schema_metric_split_id_history_hash_checks='passed',
                uses_wavlm=False,test_status='not_run')
    write_json(out/'validation_report.json',result)
    return rows,gates


def report(path):
    out=Path(path).resolve();rows,gates=validate(out)
    baseline=json.loads((out/'reused_baselines.json').read_text());neural=[r for r in rows if r['family']=='neural']
    methods=list(dict.fromkeys(r['method'] for r in baseline+neural));summary={}
    for method in methods:
        runs=[r for r in baseline+neural if r['method']==method];f1=[r['val']['macro_f1'] for r in runs]
        summary[method]=dict(mean=float(np.mean(f1)),sd=float(np.std(f1,ddof=1)),accuracy=float(np.mean([r['val']['accuracy'] for r in runs])))
    contrasts=[]
    for candidate,control in [('lmf_rank4','projected_concat_control'),('aux_supcon_intra','aux_control'),('aux_supcon_cross','aux_control')]:
        differences=[next(r['val']['macro_f1'] for r in neural if r['method']==candidate and r['seed']==s)-next(r['val']['macro_f1'] for r in neural if r['method']==control and r['seed']==s) for s in (42,43,44)]
        contrasts.append(dict(candidate=candidate,control=control,seed_differences=differences,mean_difference=float(np.mean(differences)),
            positive_seeds=sum(v>0 for v in differences),followup_gate=bool(np.mean(differences)>=.02 and sum(v>0 for v in differences)>=2)))
    write_json(out/'neural_summary.json',summary);write_json(out/'paired_contrasts.json',contrasts)
    gate_summary={}
    for condition in dict.fromkeys(r['condition'] for r in gates):
        # Average repeats WITHIN seed first; repeated permutations are not independent actor samples.
        per_seed=[]
        for seed in (42,43,44):
            group=[r for r in gates if r['condition']==condition and r['seed']==seed]
            per_seed.append(dict(seed=seed,**{k:float(np.mean([r[k] for r in group])) for k in ('macro_f1','accuracy','prediction_change_rate','mean_absolute_probability_change')}))
        gate_summary[condition]=dict(per_seed=per_seed,**{k:float(np.mean([r[k] for r in per_seed])) for k in ('macro_f1','accuracy','prediction_change_rate','mean_absolute_probability_change')})
    write_json(out/'gate_summary.json',gate_summary)
    svm_selection=json.loads((out/'svm_selection.json').read_text());winners=[next(r for r in rows if r['run_id']==rid) for rid in svm_selection['winners'].values()]
    sections=['## 1. Question and comparison','## 2. Pipeline at a glance','## 3. Results and selection',
              '## 4. Interpretation and limitations','## 5. Handoff and next step']
    for row in rows:
        folder=Path(row['folder']);m=row['method'];neural=row['family']=='neural'
        if m=='lmf_rank4':question='Does rank4 low-rank bilinear interaction improve on projected concatenation?'
        elif 'supcon' in m:question='Does the fixed supervised contrastive objective improve the auxiliary-CE residual control?'
        elif m=='aux_control':question='Can the previous auxiliary-CE residual result be reproduced under the new study wrapper?'
        else:question='How does this fixed-feature classifier/fusion configuration perform within the predeclared comparison budget?'
        config=json.loads((folder/'run.json').read_text())['pipeline']['fusion']['parameters']
        if m.startswith('aux_'):
            architecture='Frozen audio MLP176→64→8 in eval; train tanh projections176/576→64; correction MLP128→64→8, ReLU/dropout0.2, learned scalar; two auxiliary Linear64→8 heads. SupCon variants add shared Linear64→64 projection, temperature0.1/weight0.1.'
        elif neural:
            architecture='Tanh projections176/576→64; '+('append constant1; two rank4 factor tensors[4,65,64], learned rank weights/bias; ' if m=='lmf_rank4' else 'concat128→Linear64; ')+ 'LayerNorm64; MLP64→64→8, ReLU/dropout0.2.'
        else:architecture='SVC with probability=False, OVR scores/break_ties=True. '+json.dumps(row['configuration'])
        pcs=row['val']['per_class'];weak=min(pcs,key=lambda v:v['f1']);strong=max(pcs,key=lambda v:v['f1'])
        cm=np.array(row['val']['confusion_matrix_counts']);np.fill_diagonal(cm,0);i,j=np.unravel_index(cm.argmax(),cm.shape)
        notes=[f'# {row["run_id"]} — Pipeline Experiment Record v2\n','Owner/member: M5 / Jiang Yushan',
          'Structured configuration and results: [run.json](run.json)\n',sections[0],question,
          'Comparison type: fusion/classifier. Exact candidate budget and source hashes: [plan](../../plan.json). Controls: [reused baselines](../../reused_baselines.json); paired neural contrasts: [contrasts](../../paired_contrasts.json).\n',
          sections[1],'Frozen historical M1 48D + MFCC Transformer128D audio; MobileNet576D visual. No WavLM. Normalization fitted on train only; all1200 train/val IDs retained.',
          architecture,'Training: frozen features; '+('AdamW lr0.001, batch64, max30epoch/patience8; seed '+str(row['seed']) if neural else 'deterministic libsvm; no repeated-seed estimate')+'.\n',
          sections[2],f'Validation Macro-F1={row["val"]["macro_f1"]:.10f}; Accuracy={row["val"]["accuracy"]:.10f}. Test not run.',
          f'Best epoch: {row.get("best_epoch")}. Selected within SVM family: {row.get("selected_in_family",False)}. All candidates retained; no test-based selection.\n',
          sections[3],f'Lowest validation class F1: {weak["class_name"]}={weak["f1"]:.4f}; highest: {strong["class_name"]}={strong["f1"]:.4f}.',
          f'Largest off-diagonal confusion: {CLASSES[i]}→{CLASSES[j]}, {int(cm[i,j])} samples.',
          'These observations are descriptive on four validation actors. They do not establish a causal mechanism or unseen-actor generalization.',
          'LMF changes effective parameter count; SupCon changes training objective. SVM hyperparameters are validation-selected under a different search budget. Historical test was already exposed.\n',
          sections[4],'Checkpoint and SHA-256, predictions and history references are in run.json. No new encoder feature export is claimed.',
          ('predictions_val.csv contains eight-class softmax probabilities.' if neural else 'Probability columns are intentionally blank; decision_scores_val.csv contains uncalibrated OVR scores.'),
          'Class order: '+', '.join(CLASSES)+'. Final feature delivery requires a new cache, train normalizers and matching audio head; retain this historical record.']
        (folder/'notes.md').write_text('\n\n'.join(notes)+'\n')
        assert all(s in (folder/'notes.md').read_text() for s in sections)
    checks=json.loads((out/'validation_report.json').read_text());checks['notes_follow_shared_five_section_template']=42
    write_json(out/'validation_report.json',checks)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=methods+[r['family']+' (val selected)' for r in winners]
    values=[summary[m]['mean'] for m in methods]+[r['val']['macro_f1'] for r in winners]
    errors=[summary[m]['sd'] for m in methods]+[0.]*len(winners)
    fig,ax=plt.subplots(figsize=(11,7));ax.barh(range(len(names)),values,xerr=errors,capsize=3,
        color=['#85929e']*5+['#2374ab']*5+['#dd9652']*len(winners))
    ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(0,.9)
    ax.set_xlabel('Validation Macro-F1; neural error bars: sample SD across seeds 42/43/44')
    ax.set_title('Historical feature literature study: validation only\nSVM uses validation-selected hyperparameters; no repeated-seed estimate')
    for i,(v,e) in enumerate(zip(values,errors)):ax.text(v+e+.01,i,f'{v:.3f}',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'validation_comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    promoted=[r['candidate'] for r in contrasts if r['followup_gate']]
    finding=('达到预设0.02追加投入门槛的候选：'+', '.join(promoted)) if promoted else '本轮没有新神经候选达到预设的平均F1提升0.02且至少2/3seed改善门槛。'
    text=['# 文献方法验证实验\n','固定旧音频176D/视觉576D特征，未接WavLM，未评价测试集。历史测试已暴露，本轮为探索。\n',
      finding+' 结论仅限本轮固定配置，不代表这些方法普遍有效或无效。',
      f'解释方面：原gate验证F1={gate_summary["original"]["macro_f1"]:.4f}，打乱后={gate_summary["shuffle_gate"]["macro_f1"]:.4f}；平均{gate_summary["shuffle_gate"]["prediction_change_rate"]:.2%}的预测变化。模型确实依赖gate与当前输入的配对，但尚未证明gate估计的是质量。\n',
      '15次神经训练（5方法×3seed）＋27个确定性SVM候选；每个配置独立v2记录。全部保留，不只记录最佳配置。\n',
      '![验证结果](validation_comparison.png)\n','## 神经方法\n','| 方法 | 验证Macro-F1均值±SD | 验证Accuracy |','|---|---:|---:|']
    for m,s in summary.items():text.append(f'| {m} | {s["mean"]:.4f} ± {s["sd"]:.4f} | {s["accuracy"]:.4f} |')
    text+=['\n## 逐seed成对对照\n','| 方法 | 对照 | 均值F1变化 | seed42/43/44变化 | 正向seed |','|---|---|---:|---|---:|']
    for r in contrasts:text.append(f'| {r["candidate"]} | {r["control"]} | {r["mean_difference"]:+.4f} | '+', '.join(f'{v:+.4f}' for v in r['seed_differences'])+f' | {r["positive_seeds"]}/3 |')
    text+=['\nLMF rank4使用追加常数1的低秩双线性交互，前后投影、LayerNorm和MLP是本项目设置；对照采用拼接→64D线性层，二者参数量不同。',
      'SupCon权重固定0.1、temperature0.1，无额外调参。intra仅在同模态内做同情绪正样本，cross仅在另一模态中做同情绪正样本；不是将其他同类样本当负例。所有对比标签来自train。',
      'aux_control三个seed复现上轮残差+辅助CE的F1；不把参考重跑计为新方法收益。',
      '\n## SVM／核融合\n','| 家族 | 选中的配置 | 验证Macro-F1 | 验证Accuracy |','|---|---|---:|---:|']
    for r in winners:text.append(f'| {r["family"]} | {json.dumps(r["configuration"])} | {r["val"]["macro_f1"]:.4f} | {r["val"]["accuracy"]:.4f} |')
    text+=['\nSVM为确定性单次fit，不计算虚构三seed标准差。选型只使用验证集，搜索预算与神经方法不同，不能把最佳候选优势完全归因于分类器家族。',
      '所有SVM的probability=False；canonical CSV概率列留空，另存8列OVR决策分数。未把分数softmax后冒充已校准概率。核融合是固定alpha网格，不是完整SimpleMKL。',
      '\n## Gate覆盖诊断\n','固定模型和音视频输入，仅覆盖64D gate向量。shuffle在240条验证样本间重排整行gate，每seed20次，不改变输入特征；固定均值来自train。\n',
      '| 条件 | 验证F1（先平均每seed内重复） | 相对原gate预测变化率 |','|---|---:|---:|']
    for name,s in gate_summary.items():text.append(f'| {name} | {s["macro_f1"]:.4f} | {s["prediction_change_rate"]:.4f} |')
    text+=['\nlabel_oracle_extremes使用验证真值在全音频/全视觉gate两端选取，及其shuffle仅为正控制；不是可部署模型，也不是任意gate的理论上限。',
      '预测对gate敏感只能说明该组件影响决策，不能证明gate识别了输入质量或给出了因果贡献。F1近似不变时也要查看逐样本预测和概率变化，不能只凭一个汇总分数判断组件未使用。',
      '\n## 来源与复跑\n',
      'LMF：[Liu等，ACL2018](https://aclanthology.org/P18-1209/)；SupCon：[Khosla等，NeurIPS2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html)。',
      '核融合借鉴[SimpleMKL思想](https://www.jmlr.org/beta/papers/v9/rakotomamonjy08a.html)，未实现原优化器；gate覆盖借鉴[可靠性打乱诊断](https://arxiv.org/abs/2606.26473)，当前gate不是经过验证的质量分数。',
      'plan.json执行前锁定预算；runs.json、每run的checkpoint/history/预测/notes、代码快照和哈希均保留。validation_report.json独立重算42份预测及138个gate条件。',
      '最终音频交付后重新适配pooling、生成开发cache、重训匹配音频头和所有基线，并更新source_run记录；不能复用本轮旧音频head。']
    (out/'实验结果与解释.md').write_text('\n'.join(text)+'\n')
    write_json(out/'report_code_hash.json',{'literature_report.py':sha256(__file__)})
    print(json.dumps(dict(neural=summary,contrasts=contrasts,svm=[{k:r[k] for k in ('method','configuration','val')} for r in winners],
                         gate=gate_summary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True)
    report(p.parse_args().study)
