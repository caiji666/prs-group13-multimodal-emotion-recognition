"""Inference-only registry and method-specific diagnostics for final M5 checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
sys.path.insert(0,str(PROJECT/'multimodal_fusion'))
from fusion import CLASSES
from fusion.data import normalize
from fusion.final_models import FinalModel
from fusion.paths import study_rows

RESULTS=PROJECT/'multimodal_fusion/results/m5_final_delivery_20261027'
DEVELOPMENT=PROJECT/'multimodal_fusion/data/final_delivery_20261027/development.npz'
NAMES={'audio_only':'Audio-only MLP','visual_only':'Visual-only MLP','concat':'Feature concatenation',
       'gated':'Gated fusion','cross_attention':'Cross-attention','cross_bidirectional':'Bidirectional cross-attention',
       'audio_residual_aux':'Audio residual + auxiliary supervision','lmf_rank4':'Low-rank fusion (LMF)',
       'decision_equal':'Equal-weight decision fusion','decision_tuned':'Validation-weighted decision fusion'}

def clean(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy().tolist()
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    return value

class Engine:
    def __init__(self):
        torch.set_num_threads(2)
        self.rows=study_rows(RESULTS)
        self.lock=json.loads((RESULTS/'evaluation_lock.json').read_text())
        self.models={};self.raw_alpha=None

    def row(self,method,audio='learned',visual='hc'):
        return next((r for r in self.rows if r['method']==method and r['audio']==audio
                     and r['visual']==visual and r['seed']==42),None)

    def methods(self,audio='learned',visual='hc'):
        return [k for k in NAMES if (k.startswith('decision') and audio=='learned') or self.row(k,audio,visual)]

    def samples(self):
        # Only validation IDs are exposed in the demo. Test data are not loaded here.
        with np.load(DEVELOPMENT,allow_pickle=False) as z:
            return z['sample_ids'][z['splits']=='val'].tolist()

    def cached(self,sid,audio='learned',visual='hc'):
        if sid not in self.samples():raise ValueError('Replay is restricted to existing validation samples.')
        if audio not in ('learned','handcrafted','combined') or visual not in ('hc','deep','combined'):
            raise ValueError('Unknown feature representation.')
        with np.load(DEVELOPMENT,allow_pickle=False) as z:
            ix=int(np.flatnonzero(z['sample_ids']==sid)[0]);length=int(z['audio_lengths'][ix])
            akey='audio_pooled' if audio=='learned' else 'audio_'+audio+'_pooled'
            d={'audio_pooled':z[akey][ix:ix+1], 'audio_sequence':z['audio_sequence'][ix:ix+1,:length],
               'audio_lengths':np.array([length],np.int64),'audio_probs':z['audio_probs'][ix:ix+1],
               'visual_pooled':z['visual_'+visual+'_pooled'][ix:ix+1],
               'visual_sequence':z['visual_'+visual+'_sequence'][ix:ix+1],
               'visual_lengths':np.array([16],np.int64),'visual_probs':z['visual_'+visual+'_probs'][ix:ix+1]}
            meta={'source_kind':'validation_replay','sample_id':sid,'ground_truth':int(z['labels'][ix]),
                  'warnings':['Validation feature replay; no uploaded video was decoded or extracted. This sample was used for model selection and is not new generalization evidence.'],
                  'faces':[],'audio_times':[],'frame_times':[],
                  'audio':{'tokens':length,'pooled_dim':d['audio_pooled'].shape[-1]},
                  'visual_extractor':{'hc':'HC320','deep':'MobileNetV3 576D','combined':'HC+deep 896D'}[visual]}
        return d,meta

    def load(self,method,audio='learned',visual='hc'):
        key=(method,audio,visual)
        if key not in self.models:
            r=self.row(*key)
            if r is None:raise ValueError('No trained checkpoint matches this feature combination. Dimensions cannot be substituted.')
            path=Path(r['folder'])/'checkpoint.pt'
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=self.lock['checkpoint_hashes'][r['run_id']]:raise ValueError('Checkpoint SHA256 verification failed.')
            ck=torch.load(path,weights_only=True,map_location='cpu')
            model=FinalModel(ck['family'],ck['method'],ck['config'],ck['normalizers']).eval()
            model.load_state_dict(ck['state_dict'],strict=True)
            self.models[key]=(model,ck,r,digest)
        return self.models[key]

    def batch(self,d,ck):
        scaled=normalize(d,ck['normalizers'])
        return {k:torch.as_tensor(v) for k,v in scaled.items() if k in
                ('audio_pooled','audio_sequence','audio_lengths','visual_pooled','visual_sequence','visual_lengths')}

    @torch.inference_mode()
    def head(self,method,d,audio='learned',visual='hc'):
        model,ck,_,_=self.load(method,audio,visual)
        if method=='visual_only':
            # The visual-only model never reads audio; do not apply its unused
            # audio normalizer to a separately selected 48D/176D representation.
            scaled=normalize(d,{k:v for k,v in ck['normalizers'].items() if k.startswith('visual_')})
            batch={k:torch.as_tensor(v) for k,v in scaled.items() if k in ('audio_pooled','audio_sequence','audio_lengths','visual_pooled','visual_sequence','visual_lengths')}
        else:batch=self.batch(d,ck)
        return model(batch).softmax(-1)[0].numpy()

    def raw_visual_alpha(self):
        """A separate UI decision protocol: validation-only tuning for the M5 visual MLP."""
        if self.raw_alpha is None:
            a=next(r for r in self.rows if r['method']=='upstream_audio')
            v=self.row('visual_only')
            def read(r):
                with (Path(r['folder'])/'predictions_val.csv').open() as f:return {x['sample_id']:x for x in csv.DictReader(f)}
            aa,vv=read(a),read(v);ids=sorted(aa)
            if set(ids)!=set(vv):raise ValueError('Validation probability IDs mismatch')
            names=['p_'+c for c in CLASSES]
            ap=np.array([[float(aa[s][c]) for c in names] for s in ids]);vp=np.array([[float(vv[s][c]) for c in names] for s in ids])
            y=np.array([int(aa[s]['y_true']) for s in ids])
            grid=[{'alpha':round(float(x),2),'macro_f1':float(f1_score(y,(x*ap+(1-x)*vp).argmax(1),labels=list(range(8)),average='macro',zero_division=0))} for x in np.linspace(0,1,21)]
            winner=max(grid,key=lambda r:(r['macro_f1'],-abs(r['alpha']-.5)))
            self.raw_alpha={'alpha':winner['alpha'],'validation_macro_f1':winner['macro_f1'],
                            'protocol':'demo_audio_original_visual_M5_MLP_seed42_validation_tuned',
                            'validation_n':len(y),'test_used':False,'grid':grid,
                            'audio_run':a['run_id'],'visual_run':v['run_id']}
        return self.raw_alpha

    @torch.inference_mode()
    def predict(self,d,meta,method='concat',audio='learned',visual='hc'):
        if method not in self.methods(audio,visual):raise ValueError('Incompatible features and fusion method.')
        warnings=list(meta.get('warnings',[]))
        uploaded=meta['source_kind']=='uploaded_video'
        if uploaded:
            if visual!='hc':raise ValueError('Raw video currently supports only HC320 visual features.')
            d=dict(d);d['visual_probs']=self.head('visual_only',d)
            if d['visual_probs'].ndim==1:d['visual_probs']=d['visual_probs'][None]
            warnings.append('Visual probabilities use the M5 visual-only MLP (seed 42), because the upstream visual SVM is unavailable. Decision fusion follows a separate validation-weighted protocol.')
        ap=d['audio_probs'][0];vp=d['visual_probs'][0]
        explain={};provenance={};trace=[]
        if method.startswith('decision'):
            if method=='decision_equal':alpha=.5
            elif uploaded:alpha=self.raw_visual_alpha()['alpha']
            else:alpha={'hc':.6,'deep':.7,'combined':.7}[visual]
            probabilities=alpha*ap+(1-alpha)*vp
            explain={'kind':'decision','title':'Exact decision probability decomposition','alpha_audio':alpha,
                     'audio_term':(alpha*ap).tolist(),'visual_term':((1-alpha)*vp).tolist(),
                     'note':'The two terms sum exactly to the fused probability. This is an algorithmic decomposition, not a causal contribution ratio.'}
            provenance={'protocol':self.raw_visual_alpha() if uploaded and method=='decision_tuned' else 'equal' if method=='decision_equal' else 'final_delivery_decision',
                        'visual_probability_head':'M5 visual MLP seed42' if uploaded else 'delivered original probabilities'}
        else:
            model,ck,r,digest=self.load(method,audio,visual);batch=self.batch(d,ck)
            logits=model(batch);probabilities=logits.softmax(-1)[0].numpy()
            target=int(probabilities.argmax())
            provenance={'run_id':r['run_id'],'checkpoint_sha256':digest,'seed':42,
                        'selection':'fixed seed42 for the interactive demo; not a test-selected best seed',
                        'validation_macro_f1':r['val']['macro_f1']}
            # Fixed-target sensitivity on a frozen model. Lengths/masks stay valid.
            for modality in ('audio','visual'):
                changed={k:v.clone() for k,v in batch.items()}
                for key in (modality+'_pooled',modality+'_sequence'):changed[key].zero_()
                p=model(changed).softmax(-1)[0].numpy()
                trace.append({'condition':modality+'_at_train_mean','target_class':CLASSES[target],
                    'target_probability':float(p[target]),'delta_from_original':float(p[target]-probabilities[target]),
                    'prediction':CLASSES[int(p.argmax())],'probabilities':p.tolist()})
            core=model.inner
            if method=='gated':
                a,v=batch['audio_pooled'],batch['visual_pooled'];g=torch.sigmoid(core.gate(torch.cat([a,v],-1)))
                hidden=.5*torch.tanh(core.audio(a))+.5*torch.tanh(core.visual(v))
                half=core.classifier(hidden).softmax(-1)[0].numpy()
                explain={'kind':'gate','title':'64-dimensional gate for this sample','values':g[0].tolist(),
                         'mean_audio_gate':float(g.mean()),'fixed_half_probabilities':half.tolist(),
                         'note':'Values closer to 1 favor the audio projection. The mean gate is neither an audio contribution percentage nor a quality score.'}
            elif method in ('cross_attention','cross_bidirectional'):
                c=core if method=='cross_attention' else core.core
                a,v=c.audio(batch['audio_sequence']),c.visual(batch['visual_sequence'])
                vm=torch.arange(v.shape[1])[None]>=batch['visual_lengths'][:,None]
                context,weights=c.attention(a,v,v,key_padding_mask=vm,need_weights=True,average_attn_weights=False)
                pooled=c.norm(a+context).mean(1)
                if method=='cross_attention':manual=c.classifier(pooled)
                else:
                    reverse,rw=core.reverse(v,a,a,need_weights=True,average_attn_weights=False)
                    manual=core.joint_classifier(torch.cat([pooled,core.reverse_norm(v+reverse).mean(1)],-1))
                torch.testing.assert_close(manual,logits,atol=2e-5,rtol=2e-5)
                att=weights[0].mean(0).numpy();length=int(batch['audio_lengths'][0]);att=att[:length,:int(batch['visual_lengths'][0])]
                groups=np.array_split(np.arange(length),min(length,48))
                heat=np.stack([att[ix].mean(0) for ix in groups])
                times=meta.get('audio_times',[])
                explain={'kind':'attention','title':'Audio-query attention to visual frames','heatmap':heat.tolist(),
                         'row_token_centres':[float(np.mean(ix)) for ix in groups],
                         'row_time_sec':[float(np.mean(np.asarray(times)[ix])) for ix in groups] if times else [],
                         'frame_time_sec':meta.get('frame_times',[]),'heads_averaged':4,
                         'prediction_invariance_checked':True,
                         'note':'Averaged over four heads with valid audio tokens grouped into rows. Attention indicates internal associations; it does not establish causality or physical synchronization.'}
                if method=='cross_bidirectional':explain['reverse_visual_to_audio']=rw[0].mean((0,1)).tolist()
            elif method=='audio_residual_aux':
                baseline=core.base(batch)[0];correction=logits[0]-baseline
                explain={'kind':'residual','title':'Logit corrections to the frozen audio head',
                         'audio_logits':baseline.tolist(),'correction_logits':correction.tolist(),
                         'audio_probabilities':baseline.softmax(-1).tolist(),
                         'note':'The correction branch reads both audio and visual inputs. Its corrections cannot be attributed entirely to vision.'}
            else:explain={'kind':'sensitivity','title':'Input sensitivity for the original predicted target',
                          'note':'Replace one standardized modality with zero (its training mean), holding the other modality and model fixed. This may be out of distribution and is not causal attribution or a real-noise robustness test.'}
        return clean({'class_names':list(CLASSES),'method':method,'method_name':NAMES[method],
            'audio_representation':audio,'visual_representation':visual,
            'prediction':CLASSES[int(probabilities.argmax())],'probabilities':probabilities,
            'max_probability':float(np.max(probabilities)),
            'audio_probabilities':ap if audio=='learned' else None,'visual_probabilities':vp,
            'explanation':explain,'sensitivity':trace,'provenance':provenance,
            'input':meta,'warnings':warnings,'scope':'interactive inference; not a new research evaluation; probabilities are not calibrated confidence'})
