"""Replay a recorded five-experiment tuning plan and summarize validation-only results."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def summarize(root,baseline=None):
    root=Path(root); rows=[]; experiments=[]
    for run in sorted(root.glob('round_[0-9][0-9]')):
        if not (run/'status.json').exists() or json.loads((run/'status.json').read_text())['status']!='completed': continue
        report=json.loads((run/'comparison.json').read_text())
        protocol=json.loads((run/'protocol.json').read_text())
        for row in report['comparison']:
            summary=json.loads((run/row['encoder']/'summary.json').read_text())
            rows.append({'experiment':run.name,**row,'epochs_completed':summary['epochs_completed'],
                             'elapsed_seconds':summary['elapsed_seconds'],'checkpoint':str((run/row['checkpoint']).resolve())})
        experiments.append(dict(experiment=run.name,config=protocol['training'],selected_encoder=report['selected_encoder'],
                                macro_f1=report['comparison'][0]['macro_f1'],accuracy=report['comparison'][0]['accuracy']))
    if not rows: raise ValueError('No completed experiments')
    best=max(rows,key=lambda r:(r['macro_f1'],r['accuracy']))
    best_accuracy=max(rows,key=lambda r:(r['accuracy'],r['macro_f1']))
    report=dict(experiments_completed=len(experiments),models_completed=len(rows),selection_metric='validation_macro_f1',
                test_evaluated=False,best=best,highest_accuracy_at_macro_f1_selected_checkpoints=best_accuracy,
                experiments=experiments,all_models=rows)
    if baseline:
        previous=json.loads((Path(baseline)/'comparison.json').read_text())
        report['baseline']=previous['comparison']
        report['improvement_over_baseline']=dict(macro_f1=best['macro_f1']-previous['comparison'][0]['macro_f1'],
                                                accuracy=best['accuracy']-previous['comparison'][0]['accuracy'])
    (root/'suite_results.json').write_text(json.dumps(report,indent=2)+'\n')
    with (root/'suite_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines=['# Five-experiment validation results','', 'Selection: validation macro-F1. Test set not evaluated.','',
           '| Experiment | Encoder | Macro-F1 | Accuracy | Best epoch | Completed epochs |', '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['experiment']} | {r['encoder']} | {r['macro_f1']:.4%} | {r['accuracy']:.4%} | {r['best_epoch']} | {r['epochs_completed']} |" for r in rows]
    lines += ['',f"Best: {best['experiment']} / {best['encoder']}",f"Checkpoint: {best['checkpoint']}",'',
              'These are adaptively tuned, single-seed validation results; final test generalization remains unmeasured.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache',default='artifacts/tuning_augmentation_cache')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if len(plan['experiments'])!=5: parser.error('Plan must contain exactly five experiments')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    for i,entry in enumerate(plan['experiments'],1):
        command=[sys.executable,'-u','-m','ravdess_m2.train','--encoder','all','--config',entry['config'],
                 '--output',str(root/f'round_{i:02d}'),'--cache',args.cache]
        if entry.get('warm_start_best_of'):
            command += ['--warm-start-best-of',plan['baseline'],*[str(root/f'round_{j:02d}') for j in range(1,i)]]
        elif entry.get('warm_start_round'):
            command += ['--warm-start',str(root/f"round_{entry['warm_start_round']:02d}")]
        with (root/f'round_{i:02d}.log').open('w') as log:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            if process.wait()!=0: raise RuntimeError(f'Experiment {i} failed; inspect log')
        summarize(root,plan.get('baseline'))
    print(json.dumps(summarize(root,plan.get('baseline'))['best'],indent=2))

if __name__=='__main__': main()
