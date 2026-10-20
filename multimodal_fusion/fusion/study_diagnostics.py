"""Validation diagnostics: perturbations, gates, attention and error complementarity."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from . import CLASSES
from .data import normalize
from .experiment import seed_all, infer, tensor_batch, write_json
from .feature_pilot import val_metrics
from .models import FusionModel


@torch.no_grad()
def attention_probe(model,batch,uniform=False):
    """Reconstruct logits from weights and projected V; retain all residual/head layers."""
    a=model.audio(batch['audio_sequence']);v=model.visual(batch['visual_sequence'])
    valid_v=torch.arange(v.shape[1])[None,:]<batch['visual_lengths'][:,None]
    valid_a=torch.arange(a.shape[1])[None,:]<batch['audio_lengths'][:,None]
    _,weights=model.attention(a,v,v,key_padding_mask=~valid_v,need_weights=True,average_attn_weights=False)
    module=model.attention;dim=module.embed_dim;heads=module.num_heads
    values=F.linear(v,module.in_proj_weight[2*dim:],module.in_proj_bias[2*dim:])
    values=values.reshape(len(v),v.shape[1],heads,dim//heads).transpose(1,2)
    if uniform:
        used=valid_v[:,None,None,:].to(v.dtype)/batch['visual_lengths'][:,None,None,None]
        used=used.expand(len(v),heads,a.shape[1],v.shape[1])
    else:used=weights
    context=(used@values).transpose(1,2).reshape(len(a),a.shape[1],dim)
    context=module.out_proj(context)
    sequence=model.norm(a+context)
    pooled=(sequence*valid_a[...,None]).sum(1)/batch['audio_lengths'][:,None]
    logits=model.classifier(pooled)
    saliency=(weights*valid_a[:,None,:,None]).sum((1,2))/(batch['audio_lengths'][:,None]*heads)
    return logits,saliency,weights


def load_model(row):
    ck=torch.load(Path(row['folder'])/'checkpoint.pt',map_location='cpu',weights_only=True)
    model=FusionModel(ck['method'],**ck['config']).eval();model.load_state_dict(ck['state_dict'])
    return model,ck


def perturb(d,ix,condition,seed=12345,saliency=None):
    b=tensor_batch(d,ix,'cpu')
    rng=np.random.default_rng(seed)
    if condition.endswith('_zero'):
        modality=condition.split('_')[0]
        for key in (modality+'_pooled',modality+'_sequence'):b[key]=torch.zeros_like(b[key])
    elif '_noise_' in condition:
        modality,_,level=condition.split('_');level=float(level)
        for key in (modality+'_pooled',modality+'_sequence'):
            noise=torch.from_numpy(rng.normal(size=b[key].shape).astype('float32'))*level
            if key.endswith('sequence'):
                valid=torch.arange(b[key].shape[1])[None,:]<b[modality+'_lengths'][:,None]
                noise*=valid[...,None]
            b[key]=b[key]+noise
    elif condition=='visual_mismatched':
        labels=d['labels'][ix]
        choices=np.asarray([rng.choice(np.flatnonzero(labels!=l)) for l in labels])
        for key in ('visual_pooled','visual_sequence','visual_lengths'):b[key]=b[key][choices]
    elif condition.endswith('_frames'):
        for j,source in enumerate(ix):
            length=int(b['visual_lengths'][j]);count=min(4,length)
            if condition=='high_frames':selected=np.argsort(saliency[source,:length])[-count:]
            elif condition=='low_frames':selected=np.argsort(saliency[source,:length])[:count]
            else:selected=rng.choice(length,count,replace=False)
            b['visual_sequence'][j,selected]=0
    return b


def run(study,pilot):
    study=Path(study);pilot=Path(pilot);seed_all(0)
    out=study/'diagnostics';out.mkdir(exist_ok=False)
    rows=json.loads((study/'default_runs.json').read_text())
    summary=json.loads((study/'default_summary.json').read_text())
    best_fusion=max(('concat','gated','cross_attention'),key=lambda k:summary[k]['macro_f1_mean'])
    best_single=max(('audio_only','visual_only'),key=lambda k:summary[k]['macro_f1_mean'])
    with np.load(pilot/'development_features.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
    first_ck=torch.load(Path(rows[0]['folder'])/'checkpoint.pt',map_location='cpu',weights_only=True)
    d=normalize(raw,first_ck['normalizers']);val=np.flatnonzero(d['splits']=='val');y=d['labels'][val]
    prediction={};models={};conditions=['audio_zero','visual_zero','visual_mismatched',
       'audio_noise_0.25','audio_noise_0.5','audio_noise_1.0','visual_noise_0.25','visual_noise_0.5','visual_noise_1.0']
    interventions=[];gates={};gate_csv=[];attention_summary=[];attention_examples=[]
    for seed in (42,43,44):
        for method in ('audio_only','visual_only','concat','gated','cross_attention'):
            row=next(r for r in rows if r['seed']==seed and r['method']==method)
            model,ck=load_model(row);p=infer(model,d,val,'cpu')
            assert val_metrics(y,p)['confusion_matrix']==row['val']['confusion_matrix']
            prediction[(seed,method)]=p
            if method not in ('concat','gated','cross_attention'):continue
            base=val_metrics(y,p);target=p.argmax(1);original=p[np.arange(len(p)),target]
            for condition in conditions:
                changed=[]
                for j in range(0,len(val),64):
                    ix=val[j:j+64];b=perturb(d,ix,condition,seed=777+j)
                    with torch.no_grad():changed.append(model(b).softmax(-1).numpy())
                q=np.concatenate(changed);m=val_metrics(y,q)
                interventions.append({'seed':seed,'method':method,'condition':condition,'macro_f1':m['macro_f1'],
                   'macro_f1_drop':base['macro_f1']-m['macro_f1'], 'prediction_change_rate':float(np.mean(q.argmax(1)!=target)),
                   'original_predicted_class_probability_drop':float(np.mean(original-q[np.arange(len(q)),target]))})
            if method=='gated':
                values=[]
                for j in range(0,len(val),64):
                    b=tensor_batch(d,val[j:j+64],'cpu')
                    with torch.no_grad():g=torch.sigmoid(model.gate(torch.cat([b['audio_pooled'],b['visual_pooled']],-1)))
                    values.append(g.numpy())
                g=np.concatenate(values);gates[seed]=g
                for j,source in enumerate(val):
                    gate_csv.append([str(d['sample_ids'][source]),seed,int(y[j]),float(g[j].mean()),float(g[j].std())])
                for condition in ('audio_noise_1.0','visual_noise_1.0'):
                    vals=[]
                    for j in range(0,len(val),64):
                        b=perturb(d,val[j:j+64],condition,seed=777+j)
                        with torch.no_grad():gg=torch.sigmoid(model.gate(torch.cat([b['audio_pooled'],b['visual_pooled']],-1)))
                        vals.append(gg.numpy())
                    delta=(np.concatenate(vals)-g).mean(1)
                    write_json(out/f'gate_response_{condition}_s{seed}.json',
                               {'mean_gate_change':float(delta.mean()),'std_gate_change':float(delta.std(ddof=1)),
                                'positive_change_fraction':float(np.mean(delta>0))})
            if method=='cross_attention':
                saliency=np.zeros((len(d['labels']),16),np.float32);uniform=[]
                examples=[]
                for j in range(0,len(val),32):
                    ix=val[j:j+32];b=tensor_batch(d,ix,'cpu')
                    logits,s,w=attention_probe(model,b)
                    torch.testing.assert_close(logits,model(b),rtol=1e-4,atol=1e-5)
                    uniform.append(attention_probe(model,b,uniform=True)[0].softmax(-1).numpy())
                    saliency[ix]=s.numpy()
                up=np.concatenate(uniform)
                attention_summary.append({'seed':seed,'condition':'uniform_weights',
                   'macro_f1':val_metrics(y,up)['macro_f1'],'original_macro_f1':base['macro_f1'],
                   'prediction_change_rate':float(np.mean(up.argmax(1)!=target)),
                   'original_predicted_class_probability_drop':float(np.mean(original-up[np.arange(len(up)),target]))})
                for condition,repeats in [('high_frames',1),('low_frames',1),('random_frames',5)]:
                    for repeat in range(repeats):
                        changed=[]
                        for j in range(0,len(val),64):
                            b=perturb(d,val[j:j+64],condition,seed=9000+repeat*1000+j,saliency=saliency)
                            with torch.no_grad():changed.append(model(b).softmax(-1).numpy())
                        q=np.concatenate(changed)
                        attention_summary.append({'seed':seed,'condition':condition,'repeat':repeat,
                           'macro_f1':val_metrics(y,q)['macro_f1'],'original_macro_f1':base['macro_f1'],
                           'prediction_change_rate':float(np.mean(q.argmax(1)!=target)),
                           'original_predicted_class_probability_drop':float(np.mean(original-q[np.arange(len(q)),target]))})
                np.savez_compressed(out/f'attention_saliency_s{seed}.npz',sample_ids=d['sample_ids'][val],saliency=saliency[val])
                if seed==42:
                    # Deterministic first correct and first incorrect cases, no probability-based cherry-picking.
                    for category,mask in [('correct',target==y),('incorrect',target!=y)]:
                        selected=np.flatnonzero(mask)[:1]
                        for idx in selected:
                            b=tensor_batch(d,np.asarray([val[idx]]),'cpu');_,_,w=attention_probe(model,b)
                            weights=w[0,:,:int(b['audio_lengths'][0]),:].numpy()
                            np.savez_compressed(out/f'attention_{category}_s42.npz',sample_id=d['sample_ids'][val[idx]],weights=weights)
                            attention_examples.append({'category':category,'sample_id':str(d['sample_ids'][val[idx]]),
                                                       'label':int(y[idx]),'prediction':int(target[idx])})
    groups=[];repairs=[];cases=[];class_gain=[]
    for seed in (42,43,44):
        audio=prediction[(seed,'audio_only')].argmax(1);visual=prediction[(seed,'visual_only')].argmax(1)
        ac,vc=audio==y,visual==y
        group_masks={'both_correct':ac&vc,'audio_only_correct':ac&~vc,'visual_only_correct':~ac&vc,'both_wrong':~ac&~vc}
        baseline=prediction[(seed,best_single)].argmax(1);bc=baseline==y
        for method in ('concat','gated','cross_attention'):
            pred=prediction[(seed,method)].argmax(1);fc=pred==y
            for name,mask in group_masks.items():
                groups.append({'seed':seed,'method':method,'group':name,'n':int(mask.sum()),
                               'fusion_accuracy':float(np.mean(fc[mask])) if mask.any() else None})
            repairs.append({'seed':seed,'method':method,'baseline':best_single,
                 'repaired':int((~bc&fc).sum()),'harmed':int((bc&~fc).sum()),
                 'kept_correct':int((bc&fc).sum()),'still_wrong':int((~bc&~fc).sum())})
            m=val_metrics(y,prediction[(seed,method)]);bm=val_metrics(y,prediction[(seed,best_single)])
            for c in CLASSES:
                class_gain.append({'seed':seed,'method':method,'class_name':c,
                                   'f1_gain':m['per_class'][c]['f1-score']-bm['per_class'][c]['f1-score']})
        if seed==42:
            fp=prediction[(seed,best_fusion)].argmax(1);fc=fp==y
            criteria={'visual_corrects_audio':~ac&vc&fc,'audio_corrects_visual':ac&~vc&fc,
                      'fusion_harms_baseline':bc&~fc,'all_wrong':~ac&~vc&~fc}
            for category,mask in criteria.items():
                for j in np.flatnonzero(mask)[:2]:
                    cases.append({'category':category,'sample_id':str(d['sample_ids'][val[j]]),'label':int(y[j]),
                                  'audio_prediction':int(audio[j]),'visual_prediction':int(visual[j]),
                                  'fusion':best_fusion,'fusion_prediction':int(fp[j]),
                                  'audio_probabilities':prediction[(seed,'audio_only')][j].tolist(),
                                  'visual_probabilities':prediction[(seed,'visual_only')][j].tolist(),
                                  'fusion_probabilities':prediction[(seed,best_fusion)][j].tolist()})
    for name,value in [('interventions',interventions),('complementarity_groups',groups),('repair_harm',repairs),
                       ('per_class_gain',class_gain),('representative_cases',cases),('attention_controls',attention_summary),
                       ('attention_examples',attention_examples)]:write_json(out/(name+'.json'),value)
    with (out/'gates_val.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','seed','label','mean_audio_gate','std_across_dimensions']);w.writerows(gate_csv)
    write_json(out/'scope.json',{'evaluation_split':'val','test_status':'not_run',
        'best_default_fusion_by_mean':best_fusion,'strongest_matched_unimodal_by_mean':best_single,
        'upstream_probabilities_used':False,'perturbation_seed_policy':'fixed across models/seeds',
        'attention_reconstruction_matches_forward':True,
        'limitations':['Feature perturbations, not raw-audio/video degradation or causal attribution.',
                       'Gate averages are descriptive, not contribution percentages.',
                       'Validation-set descriptive explanations; not independent final test evidence.',
                       'Frame selection occurs in token space; no timestamp provenance supplied.']})
    print('Validation diagnostics complete:',out,flush=True)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True);p.add_argument('--pilot',required=True)
    a=p.parse_args();run(a.study,a.pilot)
