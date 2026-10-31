"""Audit published final test results for all48 M2 fixed checkpoints."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ravdess_m2.data import LABELS,read_splits
from ravdess_m2.train import metrics

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    files=sorted((repo/'experiment_records').glob('M2_*/run.json'));assert len(files)==48
    protocol=json.loads((repo/'experiment_records/shared/M2/final_test_20261026/protocol.json').read_text());assignment=read_splits(repo/'multimodal_fusion/configs/actor_splits.csv');testactors={k for k,v in assignment.items() if v=='test'}
    total=0
    for path in files:
        r=json.loads(path.read_text());rid=r['run_id'];assert r['test_status']=='completed' and r['run_status']=='completed'
        cp=repo/r['artifacts']['checkpoint_ref'];assert sha(cp)==r['artifacts']['checkpoint_sha256']==protocol['checkpoint_hashes_before'][rid]
        rows=list(csv.DictReader((repo/r['artifacts']['predictions_test_ref']).open()));assert len(rows)==240 and len({x['sample_id'] for x in rows})==240
        truth=[];pred=[]
        for row in rows:
            assert row['run_id']==rid and row['dataset_id']=='ravdess_speech_v1' and row['split']=='test'
            fields=row['sample_id'].split('-');assert len(fields)==6 and int(fields[-1])==int(row['actor_id']) and int(fields[1])-1==int(row['y_true'])
            assert int(row['actor_id']) in testactors
            prob=np.array([float(row['p_'+c]) for c in LABELS]);assert np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and np.isclose(prob.sum(),1,atol=1e-6)
            assert int(prob.argmax())==int(row['y_pred']);truth.append(int(row['y_true']));pred.append(int(row['y_pred']))
        m=metrics(truth,pred);record=r['metrics_by_split']['test'];assert record['n']==240 and np.array_equal(m['confusion_matrix'],record['confusion_matrix_counts'])
        assert np.array_equal(np.array(record['confusion_matrix_counts']).sum(1),[16]+[32]*7)
        for k in ('accuracy','macro_f1','macro_precision','macro_recall'):assert abs(m[k]-record[k])<1e-10
        for i,c in enumerate(record['per_class']):
            assert c['label']==i and c['class_name']==LABELS[i]
            for key,value in m['per_class'][LABELS[i]].items():assert abs(c[key]-value)<1e-10
        assert r['selection']['metric']=='macro_f1' and r['selection']['split']=='val'
        total+=len(rows)
    result=dict(valid=True,records=48,test_predictions_checked=total,checkpoint_hashes_unchanged=True,final_model=protocol['locked_final_model'],selection_basis='previous validation Macro-F1',new_training=False)
    if a.output:Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
