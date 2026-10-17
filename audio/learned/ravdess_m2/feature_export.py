"""Export and validate clean representations from the selected audio encoder."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .data import LABELS, manifest, extract, collate
from .feature_io import validate_bundle
from .models import AudioEncoder


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


class CleanDataset(Dataset):
    def __init__(self,rows,preprocessing):
        self.rows,self.preprocessing=rows,preprocessing
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        return dict(row,features=extract(row['path'],self.preprocessing,row['split'],augment=False))


def verify_source(protocol, name):
    """Allow only the documented class-name alias migration of data.py."""
    source=Path(__file__).with_name(name).read_bytes()
    expected=protocol['source_sha256'][name]
    if hashlib.sha256(source).hexdigest()==expected:
        return
    canonical=b"'disgust', 'surprise']"
    legacy=b"'disgust', 'surprised']"
    if name=='data.py' and source.count(canonical)==1 and hashlib.sha256(source.replace(canonical,legacy)).hexdigest()==expected:
        return
    raise ValueError(f'{name} differs from training; resolve compatibility before export')


def selected_checkpoint(run_dir, splits):
    run=Path(run_dir).resolve()
    status=json.loads((run/'status.json').read_text())
    selection=json.loads((run/'comparison.json').read_text())
    protocol=json.loads((run/'protocol.json').read_text())
    if status.get('status')!='completed' or status.get('diagnostic') or selection.get('diagnostic'):
        raise ValueError('Require a completed formal Stage 3 run, not diagnostic outputs')
    if selection.get('selection_metric')!='validation_macro_f1' or selection.get('test_evaluated'):
        raise ValueError('Require validation-only encoder selection')
    scores=selection['comparison']
    if {r['encoder'] for r in scores}!={'cnn','bilstm','transformer'}:
        raise ValueError('All three candidates must be compared')
    chosen=[r for r in scores if r['encoder']==selection['selected_encoder']]
    if len(chosen)!=1 or chosen[0]['macro_f1']!=max(r['macro_f1'] for r in scores):
        raise ValueError('Selected encoder does not maximize validation macro-F1')
    path=(run/selection['selected_checkpoint']).resolve()
    if not path.is_relative_to(run) or path.relative_to(run).as_posix()!=chosen[0]['checkpoint']:
        raise ValueError('Inconsistent/unsafe selected checkpoint path')
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint['encoder']!=selection['selected_encoder'] or checkpoint['epoch']!=chosen[0]['best_epoch']:
        raise ValueError('Checkpoint encoder/epoch mismatch')
    if checkpoint['provenance'].get('diagnostic') or checkpoint['class_names']!=LABELS:
        raise ValueError('Diagnostic checkpoint or class order mismatch')
    if checkpoint['provenance']['split_sha256']!=sha256(splits) or checkpoint['provenance']['protocol_sha256']!=protocol['protocol_sha256']:
        raise ValueError('Checkpoint split/protocol mismatch')
    if abs(checkpoint['validation_metrics']['macro_f1']-chosen[0]['macro_f1'])>1e-12:
        raise ValueError('Checkpoint validation score mismatch')
    for name in ('models.py','data.py'):
        verify_source(protocol,name)
    return path,checkpoint,selection,protocol


def export(run_dir,data,splits,output,batch_size=16):
    if batch_size<1: raise ValueError('batch_size must be positive')
    checkpoint_path,checkpoint,selection,protocol=selected_checkpoint(run_dir,splits)
    rows=manifest(data,splits,full=True)
    # Stage 4 never augments, trains, refits normalization, evaluates test metrics or reselects a model.
    model=AudioEncoder(checkpoint['encoder'],input_dim=checkpoint['preprocessing']['n_mfcc'])
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.eval(); model.requires_grad_(False)
    if not torch.isfinite(model.feature_mean).all() or not torch.isfinite(model.feature_std).all() or not (model.feature_std>0).all():
        raise ValueError('Invalid checkpoint normalization statistics')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    try:
        (out/'pooled').mkdir(); (out/'checkpoint').mkdir(); (out/'provenance').mkdir()
        shutil.copy2(checkpoint_path,out/'checkpoint/best.pt')
        shutil.copy2(splits,out/'actor_splits.csv')
        for name in ('comparison.json','comparison.csv','protocol.json','status.json'):
            shutil.copy2(Path(run_dir)/name,out/'provenance'/('stage3_'+name))
        shutil.copy2(Path(__file__).with_name('feature_io.py'),out/'handoff_reader.py')
        write_json(out/'preprocessing.json',checkpoint['preprocessing'])
        write_json(out/'label_mapping.json',dict(enumerate(LABELS)))
        aggregate={s:[] for s in ('train','val','test')}
        manifest_rows=[]
        loader=DataLoader(CleanDataset(rows,checkpoint['preprocessing']),batch_size=batch_size,shuffle=False,collate_fn=collate,num_workers=0)
        offset=0
        with torch.inference_mode():
            for batch in loader:
                result=model(batch['x'],batch['lengths'])
                for i,sid in enumerate(batch['sample_ids']):
                    row=rows[offset+i]
                    if sid!=row['sample_id']: raise ValueError('Export order mismatch')
                    n=int(result['lengths'][i])
                    sequence=result['sequence'][i,:n].numpy().astype(np.float32)
                    pooled=result['pooled'][i].numpy().astype(np.float32)
                    relative=Path('per_sample')/row['split']/(sid+'.npz')
                    target=out/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(target,sample_id=sid,split=row['split'],label=np.int64(row['label']),
                                        actor_id=np.int64(row['actor_id']),sequence=sequence,pooled=pooled,
                                        length=np.int64(n),padding_mask=np.zeros(n,dtype=bool))
                    aggregate[row['split']].append(dict(row,pooled=pooled))
                    manifest_rows.append(dict(sample_id=sid,actor_id=row['actor_id'],split=row['split'],label=row['label'],
                                              emotion=row['emotion'],sequence_length=n,feature_dim=sequence.shape[1],
                                              feature_file=relative.as_posix(),sha256=sha256(target)))
                offset+=len(batch['sample_ids'])
                if offset%160==0 or offset==len(rows): print(f'Exported {offset}/{len(rows)} clean recordings',flush=True)
        dim=manifest_rows[0]['feature_dim']
        for split,items in aggregate.items():
            np.savez_compressed(out/'pooled'/(split+'.npz'),sample_ids=np.array([r['sample_id'] for r in items]),
                                actor_ids=np.array([r['actor_id'] for r in items],dtype=np.int64),splits=np.array([split]*len(items)),
                                labels=np.array([r['label'] for r in items],dtype=np.int64),pooled=np.stack([r['pooled'] for r in items]),class_names=np.array(LABELS))
            with (out/'pooled'/(split+'.csv')).open('w',newline='') as f:
                writer=csv.writer(f); writer.writerow(['sample_id','actor_id','split','label',*[f'f_{j:03d}' for j in range(dim)]])
                writer.writerows([r['sample_id'],r['actor_id'],r['split'],r['label'],*r['pooled'].tolist()] for r in items)
        with (out/'manifest.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(manifest_rows[0])); writer.writeheader(); writer.writerows(manifest_rows)
        metadata=dict(schema_version=1,encoder=checkpoint['encoder'],best_epoch=checkpoint['epoch'],feature_dim=dim,
                      samples=len(rows),split_counts=dict(Counter(r['split'] for r in rows)),class_names=LABELS,
                      selected_by='validation_macro_f1',validation_macro_f1=checkpoint['validation_metrics']['macro_f1'],
                      checkpoint_sha256=sha256(checkpoint_path),split_sha256=sha256(splits),
                      protocol_sha256=protocol['protocol_sha256'],augmentation=False,normalization='checkpoint buffers fitted on clean training frames only',
                      test_features_exported=True,test_metrics_computed=False,pooled_definition='M2 learned feature only; not M1 handcrafted+learned F_audio',
                      mask_convention='True = padding; per-sample sequences are unpadded',
                      temporal_alignment='MFCC frames after edge trimming; no assumed one-to-one alignment to video frames')
        write_json(out/'metadata.json',metadata)
        report=validate_bundle(out)
        write_json(out/'validation_report.json',report)
        write_json(out/'status.json',dict(status='completed',test_metrics_computed=False))
        print(json.dumps(report,indent=2),flush=True)
        return report
    except BaseException as error:
        write_json(out/'status.json',dict(status='failed',error=repr(error)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('export','validate'))
    parser.add_argument('--run',help='Completed formal Stage 3 run directory')
    parser.add_argument('--data',default='data/raw')
    parser.add_argument('--splits',default=str(Path(__file__).resolve().parents[3]/'multimodal_fusion/configs/actor_splits.csv'))
    parser.add_argument('--output',required=True,help='Export: new directory. Validate: existing feature bundle')
    parser.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.command=='export':
        if not args.run: parser.error('--run is required for export')
        export(args.run,args.data,args.splits,args.output,args.batch_size)
    else:
        print(json.dumps(validate_bundle(args.output),indent=2))

if __name__=='__main__': main()
