"""Ten adaptive experiments; only clean training/validation hidden states are read."""
import argparse,csv,json,hashlib
from pathlib import Path
from ..prepare import ROOT
from .run import run,dump,sha

SPECS=[
 ('head_only',dict(freeze=['projection','attention','statistics'],learning_rate=.0003,mixup_alpha=0.,label_smoothing=.0,dropout=.3)),
 ('freeze_projection',dict(freeze=['projection'],learning_rate=.0001,mixup_alpha=.1,label_smoothing=.01,dropout=.3)),
 ('gentle_full_adapter',dict(freeze=[],learning_rate=.00005,mixup_alpha=.1,label_smoothing=.0,dropout=.25,weight_decay=.01)),
 ('residual_frame',dict(freeze=[],frame_depth=1,frame_width=64,learning_rate=.00015,mixup_alpha=.1,dropout=.3)),
 ('temporal_context',dict(freeze=[],time_kernel=3,learning_rate=.0001,mixup_alpha=.1,dropout=.3)),
 ('deeper_head',dict(freeze=[],head_residual=True,learning_rate=.0001,mixup_alpha=.2,dropout=.35)),
 ('freeze_head_refine_sequence',dict(freeze=['head'],learning_rate=.0001,mixup_alpha=0.,label_smoothing=.01,dropout=.25)),
 ('feature_noise_mixup',dict(freeze=[],feature_noise=.03,mixup_alpha=.35,learning_rate=.0001,dropout=.35,weight_decay=.03)),
 ('focal_classification',dict(freeze=[],feature_noise=0.,focal_gamma=1.,mixup_alpha=0.,label_smoothing=0.,learning_rate=.00005,dropout=.25)),
 ('final_conservative',dict(freeze=[],feature_noise=0.,focal_gamma=0.,mixup_alpha=.05,label_smoothing=0.,learning_rate=.00002,dropout=.2,weight_decay=.005,class_balance_power=0.))]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    baseline=ROOT/'runs/round_05';summary=json.loads((baseline/'summary.json').read_text())
    best=dict(accuracy=summary['accuracy'],macro_f1=summary['macro_f1'],checkpoint=str(baseline/'best.pt'),config=summary['config'],round=0)
    initial_sha=sha(baseline/'best.pt');results=[]
    dump(out/'protocol.json',dict(baseline=dict(accuracy=best['accuracy'],macro_f1=best['macro_f1'],checkpoint_sha256=initial_sha),selection='validation macro-F1; ties retain earlier candidate',planned_experiments=SPECS,upstream_frozen=True,test_read=False,existing_test_results_not_used=True))
    for i,(name,changes) in enumerate(SPECS,1):
        parent=best;config=dict(parent['config']);config.update(changes)
        config.update(warm_start=parent['checkpoint'],epochs=65,patience=14,batch_size=32,seed=42)
        folder=out/f'round_{i:02d}_{name}'
        dump(out/f'decision_{i:02d}.json',dict(round=i,name=name,parent_round=parent['round'],parent_checkpoint_sha256=sha(parent['checkpoint']),changes=changes,config=config,decision_basis='prior validation results only'))
        result=run(config,folder);result.update(round=i,name=name,parent_round=parent['round']);results.append(result)
        if result['macro_f1']>best['macro_f1']:best=result
        dump(out/'suite_results.json',dict(baseline_accuracy=summary['accuracy'],baseline_macro_f1=summary['macro_f1'],runs=results,best=best,test_evaluated=False))
        with (out/'suite_results.csv').open('w',newline='') as f:
            fields=['round','name','parent_round','accuracy','macro_f1','macro_precision','macro_recall','best_epoch','completed_epochs','checkpoint']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{k:x[k] for k in fields} for x in results])
        assert sha(baseline/'best.pt')==initial_sha
        print('SUITE_PROGRESS',i,'BEST',best['round'],best['accuracy'],best['macro_f1'],flush=True)
    print('TEN_ROUNDS_COMPLETED',flush=True)
if __name__=='__main__':main()
