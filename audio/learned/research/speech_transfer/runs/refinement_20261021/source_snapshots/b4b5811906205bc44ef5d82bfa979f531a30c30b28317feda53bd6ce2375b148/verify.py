"""Independent validation-only audit for completed continuation runs."""
import argparse,csv,json
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from ..prepare import ROOT
from ..run import Inputs,collate,predict,sha
from .models import build
from torch.utils.data import DataLoader

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    torch.set_num_threads(4);suite=json.loads((out/'suite_results.json').read_text());assert len(suite['runs'])==10
    index=torch.load(ROOT/'cache/pooled_index.pt',map_location='cpu',weights_only=True)
    assert {r['split'] for r in index['rows']}=={'train','val'}
    rows=[r for r in index['rows'] if r['split']=='val'];assert len(rows)==240
    actors={r['actor_id'] for r in rows};assert actors=={9,14,21,22}
    values=[torch.load(ROOT/'cache'/(r['sample_id']+'.pt'),map_location='cpu',weights_only=True)['hidden'][0].float() for r in rows]
    loader=DataLoader(Inputs(rows,values),batch_size=32,collate_fn=collate);truth=np.array([r['label'] for r in rows]);checked=0
    for result in suite['runs']:
        cp=Path(result['checkpoint']);assert sha(cp)==result['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=True)
        config=ck['config'];state=ck['state_dict'];model=build(config,state['mean'],state['std']).eval();model.load_state_dict(state)
        probabilities=predict(model,loader);pred=probabilities.argmax(1)
        csvrows=list(csv.DictReader((cp.parent/'best_val_predictions.csv').open()));assert [r['sample_id'] for r in rows]==[r['sample_id'] for r in csvrows]
        from ravdess_m2.data import LABELS
        saved=np.array([[float(r['p_'+c]) for c in LABELS] for r in csvrows]);np.testing.assert_allclose(probabilities,saved,rtol=1e-6,atol=1e-7)
        assert np.isfinite(saved).all() and np.allclose(saved.sum(1),1,atol=1e-6)
        pr,re,f1,_=precision_recall_fscore_support(truth,pred,average='macro',zero_division=0)
        for key,v in dict(accuracy=accuracy_score(truth,pred),macro_precision=pr,macro_recall=re,macro_f1=f1).items():assert abs(result[key]-v)<1e-10
        assert np.array_equal(confusion_matrix(truth,pred),result['confusion_matrix'])
        parent=torch.load(config['warm_start'],map_location='cpu',weights_only=True)['state_dict']
        for name in state:
            if any(name.startswith(prefix+'.') for prefix in config.get('freeze',[])):torch.testing.assert_close(state[name],parent[name],rtol=0,atol=0)
        torch.testing.assert_close(state['mean'],parent['mean'],rtol=0,atol=0);torch.testing.assert_close(state['std'],parent['std'],rtol=0,atol=0)
        checked+=len(rows)
    best=max(suite['runs'],key=lambda r:r['macro_f1']);assert best['checkpoint']==suite['best']['checkpoint']
    audit=dict(valid=True,rounds=10,independent_validation_predictions=checked,frozen_parameters_unchanged=True,normalization_buffers_unchanged=True,test_evaluated=False,test_audio_read=False,upstream_frozen=True,best_round=best['round'],best_accuracy=best['accuracy'],best_macro_f1=best['macro_f1'])
    (out/'verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
