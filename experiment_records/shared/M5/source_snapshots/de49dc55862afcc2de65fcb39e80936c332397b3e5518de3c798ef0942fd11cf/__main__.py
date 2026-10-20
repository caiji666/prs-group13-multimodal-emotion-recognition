import argparse
from pathlib import Path
from .data import load_pair, split_indices
from .demo import generate
from .experiment import run, predict
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description='RAVDESS frozen-feature fusion experiments')
    commands = p.add_subparsers(dest='command', required=True)
    for name in ('demo','train'):
        c = commands.add_parser(name)
        if name == 'train':
            c.add_argument('--audio', required=True)
            c.add_argument('--visual', required=True)
            c.add_argument('--synthetic', action='store_true')
        else:
            c.add_argument('--data-dir',default=str(ROOT/'data'/'synthetic'))
        c.add_argument('--output',required=True)
        c.add_argument('--split',default=str(ROOT/'configs'/'actors.json'))
        c.add_argument('--methods',nargs='+',choices=['decision','concat','gated','cross_attention'],default=['decision','concat','gated','cross_attention'])
        c.add_argument('--epochs',type=int,default=30)
        c.add_argument('--batch-size',type=int,default=64)
        c.add_argument('--lr',type=float,default=.001)
        c.add_argument('--hidden',type=int,default=64)
        c.add_argument('--heads',type=int,default=4)
        c.add_argument('--dropout',type=float,default=.2)
        c.add_argument('--patience',type=int,default=8)
        c.add_argument('--seed',type=int,default=42)
        c.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    c = commands.add_parser('validate')
    c.add_argument('--audio',required=True)
    c.add_argument('--visual',required=True)
    c.add_argument('--split',default=str(ROOT/'configs'/'actors.json'))
    c = commands.add_parser('predict')
    c.add_argument('--checkpoint',required=True)
    c.add_argument('--audio',required=True)
    c.add_argument('--visual',required=True)
    c.add_argument('--output',required=True)
    c.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    args = vars(p.parse_args())
    command = args.pop('command')
    if command == 'predict':
        args['checkpoint_path'] = args.pop('checkpoint')
        predict(**args)
    elif command == 'validate':
        d = load_pair(args['audio'],args['visual'])
        ix,_ = split_indices(d,args['split'])
        print({k:len(v) for k,v in ix.items()})
    else:
        if command == 'demo':
            args['audio'],args['visual'] = generate(args.pop('data_dir'),args['seed'])
            args['synthetic'] = True
        args['split_path'] = args.pop('split')
        report = run(**args)
        print('Selected by validation:',report['selected_by_validation'])
        print('Synthetic:',report['synthetic'])
        print('Results:',Path(args['output']).resolve())


if __name__ == '__main__':
    main()
