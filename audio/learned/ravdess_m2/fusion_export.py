"""M2 learned-only audio interface for M5's integration tests."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .data import LABELS
from .feature_io import load_pooled

AUDIO_KEY='F_audio_learned_feature_only'
PROB_KEY='P_audio_learned_feature_only'


def export_learned_audio(features,model,output):
    """Use saved clean sequence/pooled features and the frozen checkpoint MLP head."""
    features,output=Path(features),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model.eval(); model.requires_grad_(False)
    (output/'pooled').mkdir()
    with (features/'manifest.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    source={r['sample_id']:r for r in rows}
    index=[]
    for split in ('train','val','test'):
        data=load_pooled(features,split)
        with torch.inference_mode():
            logits=model.head(torch.from_numpy(data['pooled'])).numpy()
            probs=torch.from_numpy(logits).softmax(-1).numpy()
        if probs.shape!=(len(data['sample_ids']),8) or not np.isfinite(probs).all() or not np.allclose(probs.sum(-1),1,atol=1e-6):
            raise ValueError('Invalid M2 learned-only probability vectors')
        np.savez_compressed(output/'pooled'/(split+'.npz'),sample_ids=data['sample_ids'],labels=data['labels'],
                            actor_ids=data['actor_ids'],splits=data['splits'],class_names=np.array(LABELS),
                            **{AUDIO_KEY:data['pooled'],PROB_KEY:probs})
        with (output/'pooled'/(split+'.csv')).open('w',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['sample_id','actor_id','split','label',*[f'{AUDIO_KEY}_{j:03d}' for j in range(data['pooled'].shape[1])],*[f'{PROB_KEY}_{name}' for name in LABELS]])
            writer.writerows([sid,int(data['actor_ids'][i]),split,int(data['labels'][i]),*data['pooled'][i].tolist(),*probs[i].tolist()]
                             for i,sid in enumerate(data['sample_ids'].tolist()))
        for i,sid in enumerate(data['sample_ids'].tolist()):
            row=source[sid]
            with np.load(features/row['feature_file'],allow_pickle=False) as z:
                if str(z['split'])!=split or int(z['label'])!=int(data['labels'][i]):
                    raise ValueError('M5 ID/label/split alignment mismatch')
                sequence=z['sequence'].copy(); pooled=z['pooled'].copy(); n=int(z['length'])
                mask=z['padding_mask'].copy()
            np.testing.assert_array_equal(pooled,data['pooled'][i])
            np.testing.assert_allclose(sequence.mean(0),pooled,rtol=2e-5,atol=2e-6)
            target=output/'per_sample'/split/(sid+'.npz'); target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,sample_id=sid,actor_id=np.int64(row['actor_id']),split=split,label=np.int64(row['label']),
                                F_sequence=sequence,length=np.int64(n),padding_mask=mask,class_names=np.array(LABELS),
                                representation_role='M2_learned_feature_only_for_M5_testing',
                                **{AUDIO_KEY:pooled,PROB_KEY:probs[i]})
            # Read back all three outputs before considering this sample exported.
            with np.load(target,allow_pickle=False) as z:
                np.testing.assert_array_equal(z['F_sequence'],sequence)
                np.testing.assert_array_equal(z[AUDIO_KEY],pooled)
                np.testing.assert_array_equal(z[PROB_KEY],probs[i])
            index.append(dict(sample_id=sid,actor_id=int(row['actor_id']),split=split,label=int(row['label']),length=n,
                              feature_file=target.relative_to(output).as_posix()))
    with (output/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(index[0])); writer.writeheader(); writer.writerows(index)
    metadata=dict(role='M2 learned-feature-only interface for M5 functional tests',handcrafted_fusion=False,
                  final_M1_audio_outputs=False,class_names=LABELS,samples=len(index),
                  fields=dict(F_sequence='[T,128]',F_audio_learned_feature_only='[128]',P_audio_learned_feature_only='[8]'),
                  probability_definition='softmax(selected checkpoint MLP(saved pooled feature))',
                  padding_mask='True = padding; per-sample outputs are unpadded',test_metrics_computed=False,
                  training_probabilities='in-sample frozen-model outputs, not out-of-fold predictions')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (output/'README.md').write_text('''# M5 audio interface — Learned Feature Only\n\nEach `per_sample/{split}/{sample_id}.npz` contains `F_sequence [T,128]`, `F_audio_learned_feature_only [128]`, `P_audio_learned_feature_only [8]`, sample_id, actor_id, label, split, length, padding_mask and class_names. Read with np.load(path, allow_pickle=False). Pooled/probability tables are also in `pooled/{split}.npz` and CSV.\n\nF_audio_learned_feature_only is exactly M2 pooled output, without handcrafted concatenation. P_audio_learned_feature_only is softmax of the selected M2 checkpoint's own MLP head. Use these for feature/gated/decision fusion integration tests; F_sequence supplies Q tokens for cross-attention. These are explicitly learned-only baselines. Replace/compare with M1's final audio outputs once available.\n\nMatch audio/video by sample_id and verify label/split/class order. True means padding; per-sample masks are all False. Pad variable-length sequences when batching. Train downstream modules on train, choose with val, reserve test for final evaluation. Training probabilities are in-sample outputs, not out-of-fold stacking predictions. No test metrics, training or checkpoint selection is performed here.\n''')
    return metadata
