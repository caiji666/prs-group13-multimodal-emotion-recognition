"""Replay five adaptive frozen-WavLM experiments into a new directory."""
import argparse,json
from pathlib import Path
import torch
from .prepare import ROOT,main as prepare
from .run import run,dump

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    if not (ROOT/'cache/pooled_index.pt').exists():prepare()
    results=[]
    for i in range(1,6):
        config=json.loads((ROOT/f'configs/round_{i:02d}.json').read_text())
        if i==4:
            previous=torch.load(results[2]['checkpoint'],map_location='cpu',weights_only=True)
            config['layer']=[6,9,12][int(previous['state_dict']['layer_weights'].argmax())]
        if i==5:
            winner=max(results,key=lambda x:x['macro_f1']);refinement=config
            config=dict(winner['config'])
            for key in ('learning_rate','weight_decay','dropout','label_smoothing','mixup_alpha','epochs','patience'):
                config[key]=refinement[key]
            config['warm_start']=winner['checkpoint']
        result=run(config,output/f'round_{i:02d}');results.append(result)
        dump(output/'suite_results.json',dict(runs=results,best=max(results,key=lambda x:x['macro_f1']),test_evaluated=False))
    print('Best validation model:',json.dumps(max(results,key=lambda x:x['macro_f1'])))
if __name__=='__main__':main()
