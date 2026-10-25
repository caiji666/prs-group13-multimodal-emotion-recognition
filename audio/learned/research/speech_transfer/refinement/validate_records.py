"""Validate five WavLM schema-v2 records, provenance, weights and both evaluated splits."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from ..prepare import ROOT
from .models import build
from ravdess_m2.data import LABELS,read_splits
from ravdess_m2 import data

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    repo=ROOT.parents[3]
    import importlib.util
    spec=importlib.util.spec_from_file_location('legacy_validator',repo/'audio/learned/scripts/validate_experiment_records.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    template=json.loads((repo/'docs/templates/run.template.json').read_text());split=repo/'multimodal_fusion/configs/actor_splits.csv';actors=read_splits(split)
    records=sorted((repo/'experiment_records').glob('M2_encoder_wavlm_refinement_s42_20261021_*/run.json'));assert len(records)==10
    def refs(x,key=''):
        if isinstance(x,dict):
            for k,v in x.items():refs(v,k)
        elif isinstance(x,list):
            for v in x:refs(v,key)
        elif isinstance(x,str) and (key.endswith('_ref') or key=='ref') and key!='notes_ref':assert (repo/x).is_file(),x
    count=0
    for p in records:
        r=json.loads(p.read_text());legacy.schema_keys(template,r);refs(r)
        assert r['class_names']==LABELS and r['split_sha256']==sha(split)
        assert sha(repo/r['input_manifest_ref'])==r['input_manifest_sha256']
        for s in r['provenance']['source_snapshots']:assert sha(repo/s['ref'])==s['sha256']
        cp=repo/r['artifacts']['checkpoint_ref'];assert sha(cp)==r['artifacts']['checkpoint_sha256']
        ck=torch.load(cp,map_location='cpu',weights_only=True);state=ck['state_dict']
        model=build(ck['config'],state['mean'],state['std']);model.load_state_dict(state,strict=True)
        parent=torch.load(repo/r['input_artifacts'][0]['ref'],map_location='cpu',weights_only=True)['state_dict']
        for name in state:
            if any(name.startswith(prefix+'.') for prefix in ck['config'].get('freeze',[])):torch.testing.assert_close(state[name],parent[name],rtol=0,atol=0)
        assert r['test_status']=='not_run' and r['metrics_by_split']['test'] is None
        assert ck['epoch']==r['selection']['best_epoch'] and ck['split_sha256']==r['split_sha256']
        assert ck['upstream_frozen'] and r['training']['regime']=='frozen_encoder_then_classifier'
        for parent in r['input_artifacts']:assert sha(repo/parent['ref'])==parent['sha256']
        for s in ('val','test'):
            m=r['metrics_by_split'][s]
            if m is None:assert s=='test' and r['test_status']=='not_run';continue
            rows=list(csv.DictReader((repo/r['artifacts'][f'predictions_{s}_ref']).open()));assert len(rows)==240
            assert len({x['sample_id'] for x in rows})==240
            y=[];pred=[]
            for x in rows:
                assert x['run_id']==r['run_id'] and x['dataset_id']=='ravdess_speech_v1' and x['split']==s
                fields=x['sample_id'].split('-');assert len(fields)==6 and int(fields[-1])==int(x['actor_id']) and int(fields[1])-1==int(x['y_true'])
                assert actors[int(x['actor_id'])]==s
                prob=np.array([float(x['p_'+c]) for c in LABELS]);assert np.isfinite(prob).all() and (prob>=0).all() and np.isclose(prob.sum(),1,atol=1e-6)
                assert prob.argmax()==int(x['y_pred']);y.append(int(x['y_true']));pred.append(int(x['y_pred']))
            assert np.array_equal(confusion_matrix(y,pred,labels=range(8)),m['confusion_matrix_counts'])
            pr,re,f1,_=precision_recall_fscore_support(y,pred,labels=range(8),average='macro',zero_division=0)
            for k,v in dict(accuracy=accuracy_score(y,pred),macro_precision=pr,macro_recall=re,macro_f1=f1).items():
                assert abs(m[k]-v)<1e-10
                if s=='val':assert abs(ck['validation_metrics'][k]-v)<1e-10
            pr,re,f1,support=precision_recall_fscore_support(y,pred,labels=range(8),zero_division=0)
            for i,c in enumerate(m['per_class']):
                assert c['class_name']==LABELS[i] and c['support']==int(support[i])
                for k,v in [('precision',pr[i]),('recall',re[i]),('f1',f1[i])]:assert abs(c[k]-v)<1e-10
            count+=len(rows)
    result=dict(valid=True,records=10,predictions_checked=count,test_evaluated_run=None)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
