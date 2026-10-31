"""Audit and summarize the final-delivery replay without fitting additional models."""
import argparse,csv,copy,json,shutil
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,precision_recall_fscore_support,confusion_matrix
from . import CLASSES
from .feature_inputs import sha256
from .experiment import write_json
from .structure_study import now
from .paths import study_rows

ROOT=Path(__file__).resolve().parents[2]


def audit_metrics(p,m,lookup,split,rid):
    assert len(p)==len({x['sample_id'] for x in p})==240
    for x in p:
        z=lookup[x['sample_id']];assert x['split']==z['split']==split and int(x['actor_id'])==int(z['actor_id'])==int(x['sample_id'][-2:])
        assert int(x['y_true'])==int(z['label'])==int(x['sample_id'].split('-')[1])-1 and x['run_id']==rid
    y=np.array([int(x['y_true']) for x in p]);pred=np.array([int(x['y_pred']) for x in p])
    pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
    assert support.tolist()==[16]+[32]*7
    for k,v in [('accuracy',accuracy_score(y,pred)),('macro_f1',f1.mean()),('macro_precision',pr.mean()),('macro_recall',re.mean())]:assert abs(m[k]-v)<1e-12,(rid,split,k)
    assert m['confusion_matrix_counts']==confusion_matrix(y,pred,labels=range(8)).tolist()
    for i,c in enumerate(m['per_class']):
        assert c['class_name']==CLASSES[i] and c['support']==int(support[i])
        for k,v in [('precision',pr[i]),('recall',re[i]),('f1',f1[i])]:assert abs(c[k]-v)<1e-12
    return y,pred


def validate(out,rows):
    lock=json.loads((out/'evaluation_lock.json').read_text());assert lock['test_status']=='not_run'
    assert lock['plan_sha256']==sha256(out/'plan.json') and lock['decision_grid_sha256']==sha256(out/'decision_validation_grid.json')
    assert len(rows)==130 and sum(r['family'] in ('base','structure','literature') for r in rows)==93
    assert sum(r['family']=='svm' for r in rows)==27 and sum(r['family']=='reference' for r in rows)==10
    lookup={}
    for name in ('input_manifest.csv','test_manifest.csv'):
        with (out/name).open() as f:
            for r in csv.DictReader(f):assert r['sample_id'] not in lookup;lookup[r['sample_id']]=r
    assert len(lookup)==1440
    with (ROOT/'multimodal_fusion/configs/actor_splits.csv').open() as f:actors={int(r['actor_id']):r['split'] for r in csv.DictReader(f)}
    for r in lookup.values():assert actors[int(r['actor_id'])]==r['split']
    template=json.loads((ROOT/'docs/templates/run.template.json').read_text())
    def keys(r,t):
        if isinstance(t,dict):
            assert set(t)<=set(r)
            for k,v in t.items():keys(r[k],v)
        elif isinstance(t,list) and isinstance(r,list) and t:
            for x in r:keys(x,t[0])
    for row in rows:
        folder=out/'runs'/row['run_id'];record=json.loads((folder/'run.json').read_text());keys(record,template)
        assert record['test_status']=='completed' and record['selection']['config_locked_at']<record['selection']['test_first_evaluated_at']
        assert row['test_status']=='completed'
        for split in ('val','test'):
            with (folder/f'predictions_{split}.csv').open() as f:p=list(csv.DictReader(f))
            _,pred=audit_metrics(p,row[split],lookup,split,row['run_id']);audit_metrics(p,record['metrics_by_split'][split],lookup,split,row['run_id'])
            if row['family']=='svm':
                assert all(x['p_'+c]=='' for x in p for c in CLASSES)
                with (folder/f'decision_scores_{split}.csv').open() as f:s=list(csv.DictReader(f))
                assert [x['sample_id'] for x in s]==[x['sample_id'] for x in p]
                assert np.array_equal(np.array([[float(x['score_'+c]) for c in CLASSES] for x in s]).argmax(1),pred)
            else:
                q=np.array([[float(x['p_'+c]) for c in CLASSES] for x in p]);assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all()
                assert np.allclose(q.sum(1),1,atol=1e-5) and np.array_equal(q.argmax(1),pred)
        if row['family']!='reference':
            path=folder/('checkpoint.joblib' if row['family']=='svm' else 'checkpoint.pt')
            assert sha256(path)==lock['checkpoint_hashes'][row['run_id']]==record['artifacts']['checkpoint_sha256']
            assert row['checkpoint_roundtrip_verified']
        if row['family'] not in ('reference','svm'):
            h=json.loads((folder/'history.json').read_text());winner=max(h,key=lambda r:r['val_macro_f1'])
            assert winner['epoch']==row['best_epoch'] and abs(winner['val_macro_f1']-row['val']['macro_f1'])<1e-12
        assert record['input_manifest_sha256']==sha256(out/'input_manifest.csv')
    for name,digest in json.loads((out/'code_hashes.json').read_text()).items():assert sha256(out/'code'/name)==digest
    gate_rows=json.loads((out/'gate_diagnostic/results.json').read_text());assert len(gate_rows)==138
    for seed in (42,43,44):
        with np.load(out/f'gate_diagnostic/gate_predictions_s{seed}.npz',allow_pickle=False) as z:
            gp=z['probabilities'];gy=z['labels'];conditions=z['conditions'];repeats=z['repeats'];ids=z['sample_ids']
        assert gp.shape==(46,240,8) and np.isfinite(gp).all() and np.allclose(gp.sum(-1),1,atol=1e-5)
        assert all(lookup[s]['split']=='val' for s in ids) and np.array_equal(gy,np.array([int(lookup[s]['label']) for s in ids]))
        original=gp[0].argmax(1)
        for q,c,rep in zip(gp,conditions,repeats):
            item=next(r for r in gate_rows if r['seed']==seed and r['condition']==c and r['repeat']==(None if rep<0 else int(rep)))
            assert abs(item['macro_f1']-f1_score(gy,q.argmax(1),labels=range(8),average='macro',zero_division=0))<1e-12
            assert abs(item['prediction_change_rate']-(q.argmax(1)!=original).mean())<1e-12
    means={m:np.mean([r['val']['macro_f1'] for r in rows if r['family']=='base' and r['method']==m and r['visual']=='hc' and r['audio']=='learned']) for m in ('concat','gated','cross_attention')}
    assert lock['primary_method']==max(means,key=means.get)
    assert all(abs(lock['primary_validation_means'][m]-v)<1e-12 for m,v in means.items())
    result=dict(records=130,prediction_files=260,prediction_rows=62400,neural_fits=93,svm_fits=27,reference_pipelines=10,
                metric_schema_id_split_history_checkpoint_and_selection_checks='passed',gate_conditions_verified=138,test_status='completed',
                caveat='Rows repeat the same recordings across configurations; not 62400 independent recordings.')
    write_json(out/'validation_report.json',result)


def metadata_notes(out,row):
    folder=Path(row['folder']);p=folder/'run.json';r=json.loads(p.read_text());method=row['method'];family=row['family']
    ad={'learned':128,'handcrafted':48,'combined':176}[row['audio']];vd={'hc':320,'deep':576,'combined':896}[row['visual']]
    upstream=json.loads((out/'input_audit.json').read_text())['audio_metadata']
    template=json.loads((ROOT/'docs/templates/run.template.json').read_text())['pipeline']['feature_extractors'][0]
    ae=copy.deepcopy(template);ae.update(component_id='audio',modality='audio',kind='learned',method=upstream['model'],source_run_id=upstream['selection_run_id'],checkpoint_sha256=upstream['checkpoint_sha256'],trainable_in_this_run=False,output_pooled_shape=[None,128],output_sequence_shape=[None,None,128])
    ve=copy.deepcopy(template);ve.update(component_id='visual',modality='visual',kind='handcrafted' if row['visual']=='hc' else row['visual'],method='M3 supplied '+row['visual'],trainable_in_this_run=False,output_pooled_shape=[None,vd])
    r['pipeline']['feature_extractors']=[ae,ve];r['source_runs']['audio_learned']=upstream['selection_run_id']
    residual=method.startswith(('audio_residual','aux_'));reference=family=='reference';svm=family=='svm'
    if method=='audio_only':dim=ad
    elif method=='visual_only':dim=vd
    elif method=='concat':dim=ad+vd
    elif method=='audio_sequence_mean':dim=128
    elif method=='sequence_mean_concat':dim=128+vd
    elif method=='cross_bidirectional':dim=128
    elif residual:dim=None
    else:dim=64
    if residual:architecture='Frozen supplied audio MLP128→64→8, ReLU/dropout0.3 in eval; correction: projections A128/V'+str(vd)+'→64 tanh, concat128→64 ReLU/dropout0.2→8, learned scalar. Auxiliary heads64→8 when configured.'
    elif reference:architecture='No new classifier; fixed supplied probabilities and validation-only scalar mixing.'
    elif svm:architecture='SVC probability=False, OVR decision scores; '+json.dumps(row['configuration'])
    else:architecture=f'{method}; projection/interaction settings in configuration; classifier Linear({dim},64)→ReLU→Dropout(0.2)→Linear(64,8).'
    r['pipeline']['classifier'].update(input_dim=dim if not reference and not svm else None,architecture=architecture,
           type='none' if reference else 'SVC' if svm else 'frozen_audio_head_plus_residual' if residual else 'MLP')
    if svm:r['pipeline']['classifier'].update(hidden_dims=None,activation=None,dropout=None,probability_method='none')
    elif not reference:r['pipeline']['classifier']['parameters'].update(frozen_audio_head=residual,source_audio_head_sha256=json.loads((out/'input_audit.json').read_text())['audio_metadata']['checkpoint_sha256'] if residual else None)
    r['training']['other_parameters'].update(auxiliary_ce_weight_each=.2 if ('aux' in method) else 0,contrastive_weight=.1 if method.startswith('aux_supcon') else 0)
    single_audio=method in ('audio_only','audio_sequence_mean','upstream_audio') or method=='svm_audio'
    single_visual=method in ('visual_only','upstream_visual','svm_visual')
    r['pipeline']['fusion']['applied']=not(single_audio or single_visual)
    r['pipeline']['representation']['components']=[dict(component_id='audio',dim=ad,audio_kind=row['audio'])] if single_audio else [dict(component_id='visual',dim=vd,visual_kind=row['visual'])] if single_visual else [dict(component_id='audio',dim=ad,audio_kind=row['audio']),dict(component_id='visual',dim=vd,visual_kind=row['visual'])]
    if row['audio']=='handcrafted':
        r['source_runs']['audio_learned']=None;r['pipeline']['feature_extractors'][0].update(kind='handcrafted',method='M1 supplied 48D descriptors',source_run_id=None,checkpoint_sha256=None,output_pooled_shape=[None,48],output_sequence_shape=None)
    if row['audio']=='combined':
        h=copy.deepcopy(r['pipeline']['feature_extractors'][0]);h.update(component_id='audio_handcrafted',kind='handcrafted',method='M1 supplied 48D descriptors',source_run_id=None,checkpoint_sha256=None,output_pooled_shape=[None,48],output_sequence_shape=None)
        r['pipeline']['feature_extractors'].append(h)
        r['pipeline']['representation']['components']=[dict(component_id='audio',dim=128),dict(component_id='audio_handcrafted',dim=48)]+([dict(component_id='visual',dim=vd)] if not single_audio else [])
    visual_extractor=next(e for e in r['pipeline']['feature_extractors'] if e['modality']=='visual')
    visual_extractor['output_sequence_shape']=[None,16,vd]
    if row['visual']=='combined':
        visual_extractor.update(component_id='visual_deep',kind='learned',method='Supplied frozen MobileNetV3 Small',output_pooled_shape=[None,576],output_sequence_shape=[None,16,576])
        h=copy.deepcopy(visual_extractor);h.update(component_id='visual_handcrafted',kind='handcrafted',method='M3 supplied 320D handcrafted',output_pooled_shape=[None,320],output_sequence_shape=[None,16,320])
        r['pipeline']['feature_extractors'].append(h)
        r['pipeline']['representation']['components']=[e for e in r['pipeline']['representation']['components'] if e['component_id']!='visual']+[dict(component_id='visual_handcrafted',dim=320),dict(component_id='visual_deep',dim=576)]
    if reference:
        r['pipeline']['representation'].update(kind='probabilities',pooling=None,normalization=None,normalization_fit_split=None,
              components=[dict(modality='audio' if single_audio else 'visual' if single_visual else 'audio+visual',dim=8)])
    elif single_audio:r['pipeline']['representation']['kind']='handcrafted' if row['audio']=='handcrafted' else row['audio']
    elif single_visual:r['pipeline']['representation']['kind']='handcrafted' if row['visual']=='hc' else row['visual']
    if single_audio:r['pipeline']['feature_extractors']=[e for e in r['pipeline']['feature_extractors'] if e['modality']=='audio']
    if single_visual:
        r['pipeline']['feature_extractors']=[e for e in r['pipeline']['feature_extractors'] if e['modality']=='visual']
        r['source_runs']['audio_learned']=None
    if method in ('audio_sequence_mean','sequence_mean_concat'):r['pipeline']['representation']['pooling']='masked sequence mean, distinct from delivered attentive-statistics audio vector'
    r['report_metadata_enriched_at']=now();write_json(p,r)
    weak=min(row['test']['per_class'],key=lambda x:x['f1']);cm=np.array(row['test']['confusion_matrix_counts']);np.fill_diagonal(cm,0);i,j=np.unravel_index(cm.argmax(),cm.shape)
    text=f'''# {row['run_id']} — Pipeline Experiment Record v2

Owner/member: M5 / Jiang Yushan. Structured record: [run.json](run.json)

## 1. Question and comparison

Final-input replay: {method}; audio={row['audio']}, visual={row['visual']}, seed={row['seed']}. Compare within the matching representation and method family. [Locked plan](../../plan.json).

## 2. Pipeline at a glance

Frozen audio: WavLM continuation20261021 Round05, 128D attentive-statistics pooled and T×128 sequence. Audio handcrafted ablations use separately paired48D; combined176D contains NEW WavLM, not historical MFCC features. Visual={row['visual']} ({vd}D), fixed16 valid frames. Train-only standardization.

Classifier: {architecture}

## 3. Results and selection

Validation Macro-F1={row['val']['macro_f1']:.6f}, Accuracy={row['val']['accuracy']:.6f}. Test Macro-F1={row['test']['macro_f1']:.6f}, Accuracy={row['test']['accuracy']:.6f}. Best epoch={row.get('best_epoch')}. Test completed after evaluation_lock.json; no test reselection.

## 4. Interpretation and limitations

Lowest test class F1: {weak['class_name']}={weak['f1']:.4f}. Largest confusion: {CLASSES[i]}→{CLASSES[j]} ({int(cm[i,j])} samples). These are descriptive test findings, not a new tuning objective. Upstream feature selection, representation and classifier effects are distinct. Historical test was already exposed, so this replay is exploratory. Physical time alignment remains unverified; only relative token positions are used.

## 5. Handoff and next step

Checkpoint/hash, source-run identity, normalizers, history and per-split predictions are linked in run.json. No new encoder feature export is claimed. SVM probability columns are empty with separate OVR scores. Audio train probabilities are in-sample; no train stacking is fitted. Class order: {', '.join(CLASSES)}. Preserve all versions and candidates; new sources require new experiments.
'''
    (folder/'notes.md').write_text(text)


def report(path):
    out=Path(path).resolve();rows=study_rows(out);validate(out,rows)
    for r in rows:metadata_notes(out,r)
    validate(out,rows)
    groups={}
    for r in rows:
        if r['family'] in ('svm','reference'):continue
        groups.setdefault((r['family'],r['method'],r['audio'],r['visual']),[]).append(r)
    aggregate=[]
    for key,group in groups.items():
        assert len(group)==3
        item=dict(zip(('family','method','audio','visual'),key))
        for split in ('val','test'):
            values=[r[split]['macro_f1'] for r in group];item[split]=dict(macro_f1=float(np.mean(values)),sd=float(np.std(values,ddof=1)),accuracy=float(np.mean([r[split]['accuracy'] for r in group])))
        aggregate.append(item)
    write_json(out/'aggregate.json',aggregate)
    svm=[]
    for name in ('svm_audio','svm_visual','svm_concat','svm_kernel_fusion'):
        candidates=[r for r in rows if r['family']=='svm' and r['method']==name];winner=max(candidates,key=lambda x:x['val']['macro_f1']);svm.append(winner)
    write_json(out/'svm_validation_selections.json',[dict(run_id=r['run_id'],config=r['configuration'],val=r['val'],test=r['test']) for r in svm])
    main=[r for r in aggregate if r['family']=='base' and r['audio']=='learned' and r['visual']=='hc']
    selected=json.loads((out/'evaluation_lock.json').read_text())['primary_method']
    source=[r for r in rows if r['family']=='reference'];audio=next(r for r in source if r['method']=='upstream_audio')
    gate=json.loads((out/'gate_diagnostic/results.json').read_text());gate_summary={}
    for condition in dict.fromkeys(r['condition'] for r in gate):
        g=[r for r in gate if r['condition']==condition];gate_summary[condition]=dict(macro_f1=float(np.mean([r['macro_f1'] for r in g])),prediction_change_rate=float(np.mean([r['prediction_change_rate'] for r in g])))
    write_json(out/'gate_summary.json',gate_summary)
    # Test actor breakdown is descriptive and does not affect selection.
    actor_rows=[]
    for r in rows:
        if not(r['family']=='base' and r['visual']=='hc' and r['audio']=='learned' and r['method'] in ('audio_only','visual_only','concat','gated','cross_attention')):continue
        with (Path(r['folder'])/'predictions_test.csv').open() as f:p=list(csv.DictReader(f))
        for actor in (2,3,8,17):
            q=[x for x in p if int(x['actor_id'])==actor];y=[int(x['y_true']) for x in q];pred=[int(x['y_pred']) for x in q]
            actor_rows.append(dict(method=r['method'],seed=r['seed'],actor=actor,accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=range(8),average='macro',zero_division=0)))
    write_json(out/'per_actor_test.json',actor_rows)
    repair_rows=[]
    for split in ('val','test'):
        with (Path(audio['folder'])/f'predictions_{split}.csv').open() as f:ap={x['sample_id']:int(x['y_pred']) for x in csv.DictReader(f)}
        for r in rows:
            if r['family'] in ('svm','reference') or r['audio']!='learned' or r['visual']!='hc':continue
            with (Path(r['folder'])/f'predictions_{split}.csv').open() as f:predictions=list(csv.DictReader(f))
            repairs=sum(ap[x['sample_id']]!=int(x['y_true']) and int(x['y_pred'])==int(x['y_true']) for x in predictions)
            harms=sum(ap[x['sample_id']]==int(x['y_true']) and int(x['y_pred'])!=int(x['y_true']) for x in predictions)
            repair_rows.append(dict(family=r['family'],method=r['method'],seed=r['seed'],split=split,repaired=repairs,harmed=harms))
    write_json(out/'repair_harm_vs_delivered_audio.json',repair_rows)
    repair_summary={}
    for method in ('concat','gated','cross_attention'):
        repair_summary[method]={s:{k:float(np.mean([r[k] for r in repair_rows if r['family']=='base' and r['method']==method and r['split']==s])) for k in ('repaired','harmed')} for s in ('val','test')}
    write_json(out/'repair_harm_summary.json',repair_summary)
    table=lambda items:['| 方法 | 音频 | 视觉 | 验证 Macro-F1 | 测试 Macro-F1 | 测试 Accuracy |','|---|---|---|---:|---:|---:|']+[f'| {r["method"]} | {r["audio"]} | {r["visual"]} | {r["val"]["macro_f1"]:.4f} ± {r["val"]["sd"]:.4f} | {r["test"]["macro_f1"]:.4f} ± {r["test"]["sd"]:.4f} | {r["test"]["accuracy"]:.4f} |' for r in items]
    text=['# 最终交付特征：完整重跑结果（2026-10-27）\n',
      f'93组神经训练、27个SVM配置、10个固定概率流水线，共130条记录；全部配置先验证并锁定，再读取240条test特征。原测试历史已暴露，因此本轮仍是探索性重跑。验证预选主融合：**{selected}**。',
      '\n## 输入与交付分数\n',
      '音频选WavLM-only128D，来源M2_encoder_wavlm_refinement_s42_20261021_05；原分类头checkpoint哈希与交付完全一致，1200条开发概率复现通过。残差方法保留这一原头，不使用较弱重训头替代强基线。',
      f'交付音频头验证F1={audio["val"]["macro_f1"]:.4f}，测试F1={audio["test"]["macro_f1"]:.4f}。消息中的0.800/0.789没有指定本轮完整口径，不能与验证分数直接混为一项。',
      '视觉deep/combined测试F1复现了消息中的0.3661/0.4024；HC按交付概率argmax得到0.4570，与消息0.4718不同，尚未复现其原始predict标签。可能涉及SVM.predict与predict_proba的argmax不同，当前不能确定原因，不能用0.4718替换实际重算值。[scikit-learn关于分数与概率不一致的说明](https://scikit-learn.org/stable/modules/svm.html#scores-and-probabilities)',
      '视觉按上游选定handcrafted-only320D，deep576与combined896只作明确标注的表示对照。以包内概率逐样本重算；类名surprised统一为surprise。音频pooling不是sequence.mean；手工视觉mean已验证，combined序列由同帧HC+deep拼接。',
      '\n## 主输入的单模态及融合控制\n','audio_only和visual_only是在固定特征上重新训练的MLP控制，不等于交付的原分类头；原头及其概率融合单列在下一表。\n',*table(main),
      '\n## 原分类头与决策融合\n','| 方法 | 视觉表示 | 音频权重 | 验证F1 | 测试F1 |','|---|---|---:|---:|---:|']
    for r in source:text.append(f'| {r["method"]} | {r["visual"]} | {r["alpha"]:.2f} | {r["val"]["macro_f1"]:.4f} | {r["test"]["macro_f1"]:.4f} |')
    text+=['\n决策权重只用验证集0~1步长0.05选择；不训练概率stacker，因为音频train概率不是OOF。视觉train的_fusion字段按上游声明为actor-grouped OOF，此重跑未独立重训其OOF流程。',
      '\n## 视觉表示及音频handcrafted消融\n',*table([r for r in aggregate if r['family']=='base' and (r['visual']!='hc' or r['audio']!='learned')]),
      '\n## 结构与文献方法（主输入）\n',*table([r for r in aggregate if r['family']!='base']),
      '\n残差为固定WavLM音频logits+视觉修正；辅助CE和SupCon的loss只在train使用。不同输入表示/参数量/训练预算的影响不得全部归因于fusion机制。Uniform保留分配Q/K但其梯度未使用；双向attention参数更多。',
      '\n## SVM基线（每家族按验证选择，非三seed均值）\n','| 方法 | 配置 | 验证F1 | 测试F1 |','|---|---|---:|---:|']
    for r in svm:text.append(f'| {r["method"]} | {json.dumps(r["configuration"])} | {r["val"]["macro_f1"]:.4f} | {r["test"]["macro_f1"]:.4f} |')
    text+=['\n全部27个候选记录保留；无重复seed标准差。SVM probability=False，预测CSV概率列为空，另有OVR分数，未伪装为校准概率。核融合仅为固定权重网格，不是完整SimpleMKL。',
      '\n## 解释：相对交付音频头的修复与破坏\n','| 方法 | split | 修复音频错误/seed | 破坏音频正确/seed |','|---|---|---:|---:|']
    for m,group in repair_summary.items():
        for s,v in group.items():text.append(f'| {m} | {s} | {v["repaired"]:.2f} | {v["harmed"]:.2f} |')
    text+=['\n这些计数解释融合在当前数据上何时帮助、何时伤害；不是因果归因，也不用于再次调整test。',
      '\n## 验证集Gate解释\n','| Gate条件 | 验证F1 | 预测变化率 |','|---|---:|---:|']
    for k,v in gate_summary.items():text.append(f'| {k} | {v["macro_f1"]:.4f} | {v["prediction_change_rate"]:.4f} |')
    text+=['\nGate打乱只改变gate，不改变特征；每seed20个排列是诊断重复，不是独立演员样本。label-oracle只在两端gate中用真值选取，是正控制，绝不计为可部署成绩。Gate影响预测不等于已证明它识别输入质量。',
      '\n## 限制与剩余工作\n',
      '不按test结果重选方法或seed。此前查看过同一test演员，不能称全新独立测试。仅16名train、4名val、4名test演员，三seed SD不是跨划分泛化置信区间。',
      '本轮评价固定RAVDESS演员划分上的冻结特征融合。原始视频接口及独立CREMA-D试验分别见demo模块与docs/results.md；原始音视频扰动、跨演员划分统计及物理时间戳对齐未纳入本轮协议。',
      '所有旧实验保留。新代码final_inputs/final_models/final_study/final_report；每个run保留v2 JSON、五节notes、预测、history、checkpoint及哈希。evaluation_lock.json与plan.json不倒写测试前状态。']
    (out/'最终特征实验结果与解释.md').write_text('\n'.join(text)+'\n')
    summary=['# Final-delivery experiment records — 2026-10-27\n','All selection used validation; test results are descriptive, with historical test exposure disclosed.\n',
      '| Run | Family | Audio / visual | Val Macro-F1 | Test Macro-F1 | Test Accuracy |','|---|---|---|---:|---:|---:|']
    for r in rows:summary.append(f'| [{r["run_id"]}](runs/{r["run_id"]}/run.json) | {r["family"]} | {r["audio"]} / {r["visual"]} | {r["val"]["macro_f1"]:.6f} | {r["test"]["macro_f1"]:.6f} | {r["test"]["accuracy"]:.6f} |')
    (out/'summary.md').write_text('\n'.join(summary)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['run_id','family','method','audio','visual','seed','val_macro_f1','test_macro_f1','test_accuracy'])
        for r in rows:w.writerow([r['run_id'],r['family'],r['method'],r['audio'],r['visual'],r['seed'],r['val']['macro_f1'],r['test']['macro_f1'],r['test']['accuracy']])
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    names=[r['method'] for r in main];vals=[r['val']['macro_f1'] for r in main];tests=[r['test']['macro_f1'] for r in main]
    fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(main));ax.bar(x-.18,vals,.36,label='Validation',yerr=[r['val']['sd'] for r in main],capsize=3);ax.bar(x+.18,tests,.36,label='Test',yerr=[r['test']['sd'] for r in main],capsize=3)
    ax.axhline(audio['test']['macro_f1'],ls='--',color='black',label='Delivered WavLM test F1');ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Macro-F1');ax.legend();ax.set_title('Final WavLM + selected handcrafted visual; fixed actor split');fig.tight_layout();fig.savefig(out/'main_comparison.png',dpi=160);plt.close(fig)
    write_json(out/'report_code_hash.json',{'final_report.py':sha256(__file__)})
    print(json.dumps({'primary_selected':selected,'main':main,'audio_reference':audio['test']['macro_f1']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);report(p.parse_args().study)
