"""Export the shipped tuned model and package M1/M5 outputs without training."""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,help='Existing RAVDESS speech WAV root')
    parser.add_argument('--name',default='tuned_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    args=parser.parse_args()
    if Path(args.name).name!=args.name or args.name in ('.','..'):
        parser.error('--name must be a single directory name')
    data=Path(args.data).expanduser().resolve()
    features=ROOT/'artifacts'/args.name
    delivery=ROOT/'deliveries'/args.name
    if features.exists() or delivery.exists() or Path(str(delivery)+'.zip').exists():
        parser.error('Use a fresh name to preserve existing outputs')
    shared=ROOT.parents[1]/'multimodal_fusion/configs/actor_splits.csv'
    subprocess.run([sys.executable,'-m','ravdess_m2.feature_export','export','--run','models/selected_run',
                    '--data',str(data),'--splits',str(shared),'--output',str(features)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','ravdess_m2.delivery','--features',str(features),
                    '--output',str(delivery),'--zip'],cwd=ROOT,check=True)
    print(f'M1 pooled features: {delivery / "features/pooled"}')
    print(f'M1 configuration: {delivery / "audio_config.json"}')
    print(f'M5 learned-only package: {delivery / "m5_learned_feature_only"}')

if __name__=='__main__':main()
