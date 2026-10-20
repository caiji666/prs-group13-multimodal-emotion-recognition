"""Final held-out evaluation of a fixed checkpoint's exported probabilities."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,classification_report,confusion_matrix
from ravdess_m2.data import LABELS,read_splits
from .prepare import ROOT,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',default=str(ROOT.parents[3]/'Deliverables/wavlm_round05/m5_learned_feature_only'))
    a=p.parse_args();source=Path(a.delivery);meta=json.loads((source/'metadata.json').read_text())
    output=source.parent/'final_test_evaluation';output.mkdir(exist_ok=False)
    checkpoint=Path(meta['checkpoint']);checkpoint=checkpoint if checkpoint.is_absolute() else ROOT.parents[3]/checkpoint;assert sha(checkpoint)==meta['checkpoint_sha256']
    split_file=ROOT.parents[3]/'multimodal_fusion/configs/actor_splits.csv'
    assert sha(split_file)==meta['split_sha256'];splits=read_splits(split_file)
    with np.load(source/'pooled/test.npz',allow_pickle=False) as z:
        ids=z['sample_ids'].copy();y=z['labels'].copy();actors=z['actor_ids'].copy();probs=z['P_audio_learned_feature_only'].copy()
        assert z['class_names'].tolist()==LABELS and set(z['splits'].tolist())=={'test'}
    assert len(y)==240 and len(set(ids.tolist()))==240 and probs.shape==(240,8)
    assert all(splits[int(actor)]=='test' for actor in actors)
    assert np.isfinite(probs).all() and (probs>=0).all() and np.allclose(probs.sum(1),1,atol=1e-6)
    pred=probs.argmax(1);precision,recall,f1,_=precision_recall_fscore_support(y,pred,labels=list(range(8)),average='macro',zero_division=0)
    report=classification_report(y,pred,labels=list(range(8)),target_names=LABELS,output_dict=True,zero_division=0)
    matrix=confusion_matrix(y,pred,labels=list(range(8)))
    result=dict(split='test',samples=len(y),accuracy=float(accuracy_score(y,pred)),macro_f1=float(f1),macro_precision=float(precision),macro_recall=float(recall),
        correct=int((y==pred).sum()),class_names=LABELS,classification_report=report,confusion_matrix=matrix.tolist(),
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),split_sha256=sha(split_file),probabilities_sha256=sha(source/'pooled/test.npz'),
        selection_basis='Round 05 fixed using validation Macro-F1 before test evaluation',test_used_for_selection=False,training_or_tuning_performed=False)
    (output/'test_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'test_predictions.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','actor_id','split','y_true','y_pred',*[f'p_{c}' for c in LABELS]])
        w.writerows([[str(ids[i]),int(actors[i]),'test',int(y[i]),int(pred[i]),*probs[i].tolist()] for i in range(len(y))])
    with (output/'test_confusion_matrix.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['true/pred',*LABELS]);w.writerows([[name,*matrix[i].tolist()] for i,name in enumerate(LABELS)])
    with (output/'test_per_class.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['class','precision','recall','f1','support'])
        w.writerows([[c,report[c]['precision'],report[c]['recall'],report[c]['f1-score'],int(report[c]['support'])] for c in LABELS])
    (output/'README.md').write_text('Final held-out evaluation of the previously fixed WavLM Round 05 checkpoint. No training or checkpoint selection occurred. Probabilities are taken from the completed delivery export, so audio is not decoded again. Metrics are kept separate from teammate input features. Earlier experiment summaries remain historical validation-only records.\n')
    print(json.dumps({k:result[k] for k in ('samples','correct','accuracy','macro_f1','macro_precision','macro_recall')},indent=2));print(output)
if __name__=='__main__':main()
