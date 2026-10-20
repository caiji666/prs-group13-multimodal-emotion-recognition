"""Validate M2 schema-v2 records, provenance, checkpoint compatibility and metrics."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch

LABELS=['neutral','calm','happy','sad','angry','fearful','disgust','surprise']

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise ValueError(message)
def schema_keys(template,value,path=''):
    if isinstance(template,dict):
        require(isinstance(value,dict),f'{path}: expected object')
        for k,v in template.items():
            require(k in value,f'{path}: missing {k}')
            schema_keys(v,value[k],path+'/'+k)
    elif isinstance(template,list) and template and isinstance(template[0],dict):
        require(isinstance(value,list),f'{path}: expected list')
        for i,item in enumerate(value):schema_keys(template[0],item,path+f'/{i}')

def validate(repo,expected_count=33):
    repo=Path(repo).resolve();folder=repo/'experiment_records'
    template=json.loads((repo/'docs/templates/run.template.json').read_text())
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('M2_*/run.json')) if not p.parent.name.startswith('M2_encoder_wavlm_')]
    require(len(records)==expected_count,f'Expected {expected_count} M2 records, got {len(records)}')
    lookup={r['run_id']:r for r in records};require(len(lookup)==len(records),'Duplicate run IDs')
    def resolve(ref):
        p=(repo/ref).resolve();require(p.is_relative_to(repo),f'Nonportable reference: {ref}')
        require(p.is_file(),f'Missing reference: {ref}');return p
    def refs(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():refs(v,k)
        elif isinstance(value,list):
            for v in value:refs(v,key)
        elif isinstance(value,str) and (key.endswith('_ref') or key=='ref') and key!='notes_ref':resolve(value)
    counts=0
    for r in records:
        schema_keys(template,r);rid=r['run_id'];location=folder/rid
        require(r['schema_version']=='2.0' and r['member']=='M2' and r['stage']=='encoder',rid+' identity')
        require(r['run_status']=='completed' and r['test_status']=='not_run' and r['metrics_by_split']['test'] is None,rid+' test status')
        require(r['class_names']==LABELS and r['metrics_scope']=='full_pipeline',rid+' class/scope')
        refs(r);require((location/r['notes_ref']).is_file(),rid+' missing notes')
        for refkey,hashkey in [('split_ref','split_sha256'),('input_manifest_ref','input_manifest_sha256')]:require(sha(resolve(r[refkey]))==r[hashkey],rid+' '+hashkey)
        for source in r['provenance']['source_snapshots']:require(sha(resolve(source['ref']))==source['sha256'],rid+' source hash')
        for source in r['input_artifacts']:
            require(source['source_run_id'] in lookup,rid+' missing parent run')
            require(sha(resolve(source['ref']))==source['sha256'],rid+' parent hash')
        artifacts=r['artifacts'];cp=resolve(artifacts['checkpoint_ref'])
        require(sha(cp)==artifacts['checkpoint_sha256'],rid+' checkpoint hash')
        ck=torch.load(cp,map_location='cpu',weights_only=True)
        require(ck['class_names']==LABELS and not ck['provenance']['diagnostic'],rid+' checkpoint classes')
        require(ck['epoch']==r['selection']['best_epoch'],rid+' checkpoint epoch')
        require(ck['provenance']['split_sha256']==r['split_sha256'],rid+' checkpoint split')
        require(ck['encoder'] in ('cnn','bilstm','transformer'),rid+' encoder')
        rows=list(csv.DictReader(resolve(artifacts['predictions_val_ref']).open()))
        require(len(rows)==240 and len({x['sample_id'] for x in rows})==240,rid+' validation IDs')
        cm=np.zeros((8,8),dtype=int)
        for x in rows:
            require(x['run_id']==rid and x['dataset_id']=='ravdess_speech_v1' and x['split']=='val',rid+' prediction identity')
            fields=x['sample_id'].split('-');require(len(fields)==6 and int(fields[-1])==int(x['actor_id']) and int(fields[1])-1==int(x['y_true']),rid+' sample ID')
            prob=np.array([float(x['p_'+name]) for name in LABELS]);require(np.isfinite(prob).all() and (prob>=0).all() and (prob<=1).all() and abs(prob.sum()-1)<=1e-5,rid+' probability')
            require(int(np.argmax(prob))==int(x['y_pred']),rid+' argmax')
            cm[int(x['y_true']),int(x['y_pred'])]+=1
        val=r['metrics_by_split']['val'];require(val['n']==240 and np.array_equal(cm,np.array(val['confusion_matrix_counts'])),rid+' confusion matrix')
        support=cm.sum(1);require(np.array_equal(support,[16]+[32]*7),rid+' class supports')
        tp=cm.diagonal();prec=np.divide(tp,cm.sum(0),out=np.zeros(8),where=cm.sum(0)>0);rec=tp/support
        f1=np.divide(2*prec*rec,prec+rec,out=np.zeros(8),where=prec+rec>0)
        values={'accuracy':float(tp.sum()/240),'macro_precision':float(prec.mean()),'macro_recall':float(rec.mean()),'macro_f1':float(f1.mean())}
        for name,v in values.items():
            require(abs(v-val[name])<1e-10,rid+' metric '+name)
            require(abs(v-ck['validation_metrics'][name])<1e-10,rid+' checkpoint metric '+name)
        for i,entry in enumerate(val['per_class']):
            require(entry['label']==i and entry['class_name']==LABELS[i] and entry['support']==int(support[i]),rid+' per-class identity')
            for k,v in [('precision',prec[i]),('recall',rec[i]),('f1',f1[i])]:require(abs(entry[k]-v)<1e-10,rid+' per-class '+k)
        history=list(csv.DictReader(resolve(artifacts['history_ref']).open()));require(len(history)==r['training']['completed_epochs'],rid+' history length')
        best=next(x for x in history if int(x['epoch'])==r['selection']['best_epoch'])
        require(abs(float(best['val_macro_f1'])-val['macro_f1'])<1e-10,rid+' best epoch score')
        require(not r['pipeline']['fusion']['applied'] and r['training']['regime']=='joint_training',rid+' training regime')
        require(r['pipeline']['classifier']['architecture']=='Linear(128,64) → ReLU → Dropout → Linear(64,8)',rid+' head')
        counts+=len(rows)
    for rid,r in lookup.items():
        seen={rid};p=r['source_runs']['audio_learned']
        while p:
            require(p in lookup and p not in seen,rid+' parent graph cycle/missing')
            seen.add(p);p=lookup[p]['source_runs']['audio_learned']
    selection=json.loads((repo/'audio/learned/models/selection.json').read_text());winner=lookup[selection['run_id']]
    require(selection['run_id']=='M2_encoder_transformer_s7_20261018_04','Wrong default model')
    require(sha(repo/'audio/learned/models/best_audio_encoder.pt')==winner['artifacts']['checkpoint_sha256'],'Default model differs from selected run')
    require(sha(repo/'audio/learned/models/selected_run/transformer/best.pt')==winner['artifacts']['checkpoint_sha256'],'Selected-run export differs')
    require(max(r['metrics_by_split']['val']['macro_f1'] for r in records)==winner['metrics_by_split']['val']['macro_f1'],'Selected winner is not best validation F1')
    return {'valid':True,'records':len(records),'groups':len({r['comparison_group'] for r in records}),'validation_predictions_checked':counts,'checkpoints_checked':len(records),'all_test_status':'not_run','selected_run_id':selection['run_id']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--expected-count',type=int,default=33);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=validate(args.repo,args.expected_count)
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
