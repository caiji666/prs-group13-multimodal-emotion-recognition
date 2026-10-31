"""Train/validation-only frozen-feature fusion; upstream probabilities are never loaded."""
import argparse
import copy
import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import sklearn
import torch
from . import CLASSES
from .data import fit_normalizers, normalize
from .experiment import metrics, seed_all, tensor_batch, infer, write_json
from .feature_inputs import load_development_features, sha256
from .models import FusionModel
ROOT = Path(__file__).resolve().parents[2]


def val_metrics(y,p):
    m=metrics(y,p)
    m.update(n=len(y),macro_precision=m['per_class']['macro avg']['precision'],
             macro_recall=m['per_class']['macro avg']['recall'])
    return m


def record_run(folder,method,row,settings,audit,output,started,environment):
    template=ROOT/'docs/templates/run.template.json'
    record=json.loads(template.read_text()) if template.exists() else {}
    m=row['val']
    record.update(run_id=f'M5_fusion_{method}_s{settings["seed"]}_20261019_01',member='M5',
                  owner='Jiang Yushan',stage='fusion',comparison_group='mobilenet_audio176_pilot_v1',
                  pipeline_name=f'frozen audio + MobileNet features -> {method} -> MLP',
                  run_status='completed',test_status='not_run',started_at=started,
                  finished_at=datetime.now().astimezone().isoformat(),seed=settings['seed'],
                  class_names=list(CLASSES),metrics_scope='full_pipeline',
                  counts={'expected':{'train':960,'val':240,'test':240},'actual':{'train':960,'val':240,'test':None}},
                  environment=environment,source_hashes=audit['source_hashes'],
                  source_status='Upstream run IDs/checkpoints not supplied; frozen supplied exports used',
                  command=' '.join(sys.argv),code_snapshot_ref=str(output/'code'))
    record['split_sha256']=audit['source_hashes']['actor_csv']
    record['input_manifest_ref']=str(output/'input_manifest.csv')
    record['input_manifest_sha256']=sha256(output/'input_manifest.csv')
    per=[{'label':i,'class_name':c,'precision':m['per_class'][c]['precision'],
          'recall':m['per_class'][c]['recall'],'f1':m['per_class'][c]['f1-score'],
          'support':int(m['per_class'][c]['support'])} for i,c in enumerate(CLASSES)]
    record['metrics_by_split']={'val':{'n':240,'accuracy':m['accuracy'],
           'macro_precision':m['macro_precision'],'macro_recall':m['macro_recall'],
           'macro_f1':m['macro_f1'],'per_class':per,'confusion_matrix_counts':m['confusion_matrix']},'test':None}
    record['training']={'regime':'frozen_encoder_then_classifier','optimizer':'AdamW',
           'learning_rate':settings['lr'],'batch_size':settings['batch_size'],
           'max_epochs':settings['epochs'],'completed_epochs':row['completed_epochs'],
           'early_stopping_patience':settings['patience'],'class_weighting':'none'}
    record['selection']={'metric':'macro_f1','split':'val','best_epoch':row['best_epoch'],
                         'selected_for_next_stage':None,'test_first_evaluated_at':None,
                         'tie_break_rule':'retain earlier epoch','selection_reason':'Single-seed pilot only'}
    record['pipeline']={'feature_extractors':[
        {'component_id':'audio_supplied','method':'M1 48D + M2 learned128D; seq128D',
         'trainable_in_this_run':False,'source_run_id':None},
        {'component_id':'visual_supplied','method':'MobileNetV3-Small (supplier filename)',
         'trainable_in_this_run':False,'source_run_id':None}],
        'representation':{'normalization':'per-dimension standardization','normalization_fit_split':'train',
                          'audio_pooled_dim':176,'visual_pooled_dim':576,'audio_sequence_dim':128,
                          'visual_sequence_dim':576,'visual_tokens_assumed_valid':16},
        'fusion':{'applied':True,'method':method,'parameters':settings,'trainable_in_this_run':True},
        'classifier':{'type':'MLP','input_dim':752 if method=='concat' else settings['hidden'],
                      'hidden_dims':[settings['hidden']],'output_dim':8,'activation':'ReLU',
                      'dropout':settings['dropout'],'probability_method':'softmax','trainable_in_this_run':True}}
    record['artifacts']={'predictions_val_ref':str(folder/'predictions_val.csv'),'predictions_test_ref':None,
                         'checkpoint_ref':str(folder/'checkpoint.pt'),'checkpoint_sha256':sha256(folder/'checkpoint.pt'),
                         'history_ref':str(folder/'history.json'),'original_metrics_ref':str(folder/'metrics.json')}
    record['limitations']=audit['notes']
    write_json(folder/'run.json',record)


def run(audio_csv, audio_sequences, visual_npz, actor_csv, output, epochs=30,
        patience=8, seed=42, batch_size=64, hidden=64, heads=4, dropout=.2, lr=.001):
    if min(epochs,patience,batch_size,hidden,heads)<1 or hidden%heads or lr<=0 or not 0<=dropout<1:
        raise ValueError('Invalid training parameters')
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True,exist_ok=True)
    started=datetime.now().astimezone().isoformat();clock=time.monotonic()
    raw,audit=load_development_features(audio_csv,audio_sequences,visual_npz,actor_csv)
    write_json(output/'adapter_audit.json',audit)
    np.savez_compressed(output/'development_features.npz',**raw)
    with (output/'input_manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','actor_id','split','label'])
        w.writerows(zip(raw['sample_ids'],raw['actors'],raw['splits'],raw['labels']))
    train=np.flatnonzero(raw['splits']=='train');val=np.flatnonzero(raw['splits']=='val')
    stats=fit_normalizers(raw,train)
    d=normalize(raw,stats)
    config=dict(audio_dim=176,visual_dim=576,audio_seq_dim=128,visual_seq_dim=576,
                hidden=hidden,heads=heads,dropout=dropout)
    settings=dict(epochs=epochs,patience=patience,seed=seed,batch_size=batch_size,lr=lr,
                  optimizer='AdamW',weight_decay=1e-4,gradient_clip=1.,device='cpu',**config)
    report={'scope':'Real frozen-feature fusion pilot, train/validation only',
            'started_at':started,'test_status':'not_run','test':None,'decision_fusion':'deferred',
            'seed':seed,'settings':settings,'counts':audit['counts'],'class_names':list(CLASSES),
            'environment':{'python':platform.python_version(),'torch':str(torch.__version__),
                           'numpy':np.__version__,'sklearn':sklearn.__version__,'device':'cpu'},
            'assumptions':audit['notes'],'source_hashes':audit['source_hashes'],'models':{}}
    snapshot=output/'code';snapshot.mkdir()
    for name in ('__init__.py','feature_inputs.py','feature_pilot.py','models.py','data.py','experiment.py'):
        source=Path(__file__).with_name(name)
        (snapshot/name).write_bytes(source.read_bytes())
    write_json(output/'code_hashes.json',{p.name:sha256(p) for p in snapshot.iterdir()})
    for method in ('concat','gated','cross_attention'):
        seed_all(seed)
        folder=output/method;folder.mkdir()
        model=FusionModel(method,**config)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        best=-1.;best_state=None;stale=0;history=[];t0=time.monotonic()
        for epoch in range(1,epochs+1):
            model.train();loss_sum=0.
            for j in range(0,len(train),batch_size):
                if j==0:
                    order=np.random.permutation(train)
                ix=order[j:j+batch_size]
                opt.zero_grad()
                logits=model(tensor_batch(d,ix,'cpu'))
                loss=torch.nn.functional.cross_entropy(logits,torch.as_tensor(d['labels'][ix],dtype=torch.long))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss in '+method)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                loss_sum+=float(loss.detach())*len(ix)
            p=infer(model,d,val,'cpu',batch_size)
            score=val_metrics(d['labels'][val],p)
            history.append({'epoch':epoch,'train_loss':loss_sum/len(train),
                            'val_macro_f1':score['macro_f1'],'val_accuracy':score['accuracy']})
            if score['macro_f1']>best+1e-8:
                best=score['macro_f1'];best_epoch=epoch;stale=0
                best_state=copy.deepcopy(model.state_dict())
            else:
                stale+=1
            print(f'{method} epoch={epoch} train_loss={loss_sum/len(train):.4f} val_f1={score["macro_f1"]:.4f}',flush=True)
            if stale>=patience:
                break
        model.load_state_dict(best_state)
        probs=infer(model,d,val,'cpu',batch_size)
        checkpoint={'method':method,'config':config,'state_dict':best_state,'normalizers':stats,
                    'class_names':list(CLASSES),'best_epoch':best_epoch,'seed':seed,
                    'test_status':'not_run','source_hashes':audit['source_hashes'],
                    'input_scope':'development train/val only','assumptions':audit['notes']}
        torch.save(checkpoint,folder/'checkpoint.pt')
        ck=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        restored=FusionModel(ck['method'],**ck['config']);restored.load_state_dict(ck['state_dict'])
        np.testing.assert_allclose(infer(restored,d,val,'cpu',batch_size),probs,rtol=1e-5,atol=1e-6)
        m=val_metrics(d['labels'][val],probs)
        row={'best_epoch':best_epoch,'completed_epochs':len(history),'val':m,'test':None,
             'test_status':'not_run','parameters':sum(p.numel() for p in model.parameters()),
             'training_seconds':time.monotonic()-t0,'checkpoint_roundtrip_verified':True}
        report['models'][method]=row
        write_json(folder/'metrics.json',row);write_json(folder/'history.json',history)
        with (folder/'predictions_val.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['sample_id','actor_id','split','label','prediction',*CLASSES])
            for i,p in zip(val,probs):
                w.writerow([d['sample_ids'][i],int(d['actors'][i]),'val',int(d['labels'][i]),int(p.argmax()),*p.tolist()])
        record_run(folder,method,row,settings,audit,output,started,report['environment'])
        write_json(output/'validation_metrics.json',report)
        print(f'{method} BEST epoch={best_epoch} val_macro_f1={best:.4f}',flush=True)
    report['selected_by_validation']=max(report['models'],key=lambda k:report['models'][k]['val']['macro_f1'])
    report['elapsed_seconds']=time.monotonic()-clock
    report['finished_at']=datetime.now().astimezone().isoformat()
    write_json(output/'validation_metrics.json',report)
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','seed','val_macro_f1','val_accuracy','best_epoch','completed_epochs','test_status'])
        for name,r in report['models'].items():
            w.writerow([name,seed,r['val']['macro_f1'],r['val']['accuracy'],r['best_epoch'],r['completed_epochs'],'not_run'])
    print('Finished. Validation-selected:',report['selected_by_validation'],'Test: not_run',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio-csv',default=str(ROOT/'features/Audio_Handoff/features/audio_fused_176d.csv'))
    p.add_argument('--audio-sequences',default=str(ROOT/'features/Audio_dl_feature_only/per_sample'))
    p.add_argument('--visual-npz',default=str(ROOT/'features/mobilenetv3_small_features.npz'))
    p.add_argument('--actor-csv',default=str(ROOT/'multimodal_fusion/configs/actor_splits.csv'))
    p.add_argument('--output',required=True)
    for name,default in [('epochs',30),('patience',8),('seed',42),('batch-size',64),('hidden',64),('heads',4)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--dropout',type=float,default=.2)
    p.add_argument('--lr',type=float,default=.001)
    run(**vars(p.parse_args()))
