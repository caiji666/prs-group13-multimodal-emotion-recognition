"""Final inference-only test evaluation of all fixed M2 experiment checkpoints."""
import argparse,csv,json,hashlib,sys,datetime
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ravdess_m2.data import LABELS,manifest,extract
from ravdess_m2.models import AudioEncoder
from ravdess_m2.train import metrics
from research.speech_transfer.models import build as initial_build
from research.speech_transfer.refinement.models import build as refinement_build
from research.speech_transfer.prepare import ROOT,LAYERS,REVISION,wave

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write_csv(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def standardized(m):
    return dict(n=240,**{k:m[k] for k in ('accuracy','macro_precision','macro_recall','macro_f1')},per_class=[dict(label=i,class_name=c,**m['per_class'][c]) for i,c in enumerate(LABELS)],confusion_matrix_counts=m['confusion_matrix'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output).resolve();output.mkdir(parents=True,exist_ok=False)
    repo=ROOT.parents[3];records=sorted((repo/'experiment_records').glob('M2_*/run.json'));assert len(records)==48
    split=repo/'multimodal_fusion/configs/actor_splits.csv';allrows=manifest(ROOT.parents[1]/'data/raw',split)
    test=[r for r in allrows if r['split']=='test'];val=[r for r in allrows if r['split']=='val']
    assert len(test)==240 and len({r['actor_id'] for r in test})==4
    checkval=[next(r for r in val if r['label']==c) for c in range(8)]
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    locked={json.loads(p.read_text())['run_id']:sha(repo/json.loads(p.read_text())['artifacts']['checkpoint_ref']) for p in records}
    dump(output/'protocol.json',dict(checkpoint_hashes_before=locked,locked_final_model='M2_encoder_wavlm_refinement_s42_20261021_05',selection_basis='previous validation Macro-F1; not changed using test scores',training=False,augmentation=False,normalization_refit=False,model_selection=False,shared_split_sha256=sha(split),test_actors=sorted({r['actor_id'] for r in test}),records=48))
    mfcc={};hidden={};upstream=None;processor=None;results=[]
    def mfcc_values(rows,config):
        key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        for row in rows:
            k=(key,row['sample_id'])
            if k not in mfcc:mfcc[k]=extract(row['path'],config,row['split'],augment=False)
        return [mfcc[(key,r['sample_id'])] for r in rows]
    def ssl_values(rows):
        nonlocal upstream,processor
        for i,row in enumerate(rows):
            sid=row['sample_id']
            if sid in hidden:continue
            if row['split']=='val':
                cached=torch.load(ROOT/'cache'/(sid+'.pt'),map_location='cpu',weights_only=True)
                assert cached['split']=='val' and cached['audio_sha256']==sha(row['path'])
                hidden[sid]=cached['hidden'].float();continue
            if upstream is None:
                from transformers import WavLMModel,Wav2Vec2FeatureExtractor
                upstream=WavLMModel.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True).eval().requires_grad_(False)
                processor=Wav2Vec2FeatureExtractor.from_pretrained(ROOT/'assets/wavlm-base-plus',local_files_only=True)
            inputs=processor(wave(row['path']),sampling_rate=16000,return_tensors='pt')
            with torch.inference_mode():o=upstream(**inputs,output_hidden_states=True)
            hidden[sid]=torch.stack([o.hidden_states[j][0] for j in LAYERS]).half().float()
            if (i+1)%50==0:print('Test upstream inference',i+1,'/',len(rows),flush=True)
        return [hidden[r['sample_id']] for r in rows]
    @torch.inference_mode()
    def predict(model,values,mfcc_model=False):
        probs=[]
        for start in range(0,len(values),16):
            xs=values[start:start+16]
            if xs[0].shape==torch.Size([3,1536]):x=torch.stack(xs);lengths=None
            else:
                lengths=torch.tensor([len(v) for v in xs]);x=torch.nn.utils.rnn.pad_sequence(xs,batch_first=True)
            logits=model(x,lengths);logits=logits['logits'] if mfcc_model else logits
            probs.append(logits.softmax(-1).numpy())
        return np.concatenate(probs)
    def probabilities(record,ck,rows):
        if 'model_state_dict' in ck:
            model=AudioEncoder(ck['encoder'],input_dim=ck['preprocessing']['n_mfcc']).eval().requires_grad_(False)
            model.load_state_dict(ck['model_state_dict'],strict=True);values=mfcc_values(rows,ck['preprocessing'])
            return predict(model,values,True)
        assert ck['upstream']['revision']==REVISION and ck['upstream']['sha256']==sha(ROOT/'assets/wavlm-base-plus/pytorch_model.bin')
        config=ck['config'];state=ck['state_dict'];builder=refinement_build if 'wavlm_refinement' in record['run_id'] else initial_build
        model=builder(config,state['mean'],state['std']).eval().requires_grad_(False);model.load_state_dict(state,strict=True)
        states=ssl_values(rows)
        if config['pooling']=='attentive_statistics':values=[h[LAYERS.index(config['layer'])] for h in states]
        else:values=[torch.cat([h.mean(1),h.std(1,unbiased=False)],-1) for h in states]
        return predict(model,values)
    for i,path in enumerate(records,1):
        r=json.loads(path.read_text());rid=r['run_id'];cp=repo/r['artifacts']['checkpoint_ref'];assert sha(cp)==locked[rid]==r['artifacts']['checkpoint_sha256'] and r['split_sha256']==sha(split)
        if r['test_status']=='completed' and r['metrics_by_split']['test'] is not None:
            results.append(dict(run_id=rid,status='preserved_existing',**{k:r['metrics_by_split']['test'][k] for k in ('accuracy','macro_f1','macro_precision','macro_recall')}));continue
        ck=torch.load(cp,map_location='cpu',weights_only=True);assert ck['class_names']==LABELS
        valprobs=probabilities(r,ck,checkval);saved={x['sample_id']:x for x in csv.DictReader((repo/r['artifacts']['predictions_val_ref']).open())}
        expected=np.array([[float(saved[row['sample_id']]['p_'+c]) for c in LABELS] for row in checkval]);np.testing.assert_allclose(valprobs,expected,rtol=5e-5,atol=5e-6)
        reuse=repo/'Deliverables/wavlm_round05'
        meta_path=reuse/'m5_learned_feature_only/metadata.json'
        reused=False
        if rid=='M2_encoder_wavlm_refinement_s42_20261021_05' and meta_path.is_file():
            meta=json.loads(meta_path.read_text());assert meta['checkpoint_sha256']==locked[rid] and meta['split_sha256']==sha(split)
            with np.load(reuse/'m5_learned_feature_only/pooled/test.npz',allow_pickle=False) as z:
                lookup={str(s):j for j,s in enumerate(z['sample_ids'])};probs=np.stack([z['P_audio_learned_feature_only'][lookup[row['sample_id']]] for row in test])
                assert all(int(z['labels'][lookup[row['sample_id']]])==row['label'] for row in test)
            reused=True
        else:probs=probabilities(r,ck,test)
        assert probs.shape==(240,8) and np.isfinite(probs).all() and np.allclose(probs.sum(1),1,atol=1e-6)
        m=metrics([row['label'] for row in test],probs.argmax(1));m['n']=240
        dest=path.parent;fields=['run_id','dataset_id','sample_id','actor_id','split','y_true','y_pred']+['p_'+c for c in LABELS]
        predictions=[dict(run_id=rid,dataset_id=r['dataset_id'],sample_id=row['sample_id'],actor_id=row['actor_id'],split='test',y_true=row['label'],y_pred=int(prob.argmax()),**{'p_'+c:float(prob[j]) for j,c in enumerate(LABELS)}) for row,prob in zip(test,probs)]
        write_csv(dest/'predictions_test.csv',predictions,fields)
        with (dest/'test_confusion_matrix.csv').open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['true/pred',*LABELS]);w.writerows([[c,*m['confusion_matrix'][j]] for j,c in enumerate(LABELS)])
        write_csv(dest/'test_per_class.csv',[dict(class_name=c,**m['per_class'][c]) for c in LABELS],['class_name','precision','recall','f1','support'])
        dump(dest/'test_metrics.json',dict(**m,checkpoint_ref=r['artifacts']['checkpoint_ref'],checkpoint_sha256=locked[rid],split_sha256=sha(split),evaluated_on='2026-10-26',training_or_tuning_performed=False,test_used_for_selection=False,reused_locked_delivery_probabilities=reused))
        r['test_status']='completed';r['metrics_by_split']['test']=standardized(m);r['artifacts']['predictions_test_ref']=(dest/'predictions_test.csv').relative_to(repo).as_posix();r['artifacts']['test_metrics_ref']=(dest/'test_metrics.json').relative_to(repo).as_posix()
        r.setdefault('provenance',{})['final_test_evaluation']=dict(date='2026-10-26',checkpoint_sha256=locked[rid],validation_compatibility_samples_checked=8,training_or_tuning_performed=False,normalization_refitted=False,test_used_for_selection=False,reused_delivery_probabilities=reused)
        dump(path,r)
        with (dest/'notes.md').open('a') as f:f.write('\nFinal test evaluation added2026-10-26 after model development ended; fixed checkpoint, no fitting/augmentation/selection. Historical status/summary files retain their training-time validation-only state. Current run.json and test_metrics.json hold final test results.\n')
        results.append(dict(run_id=rid,status='imported_existing_delivery' if reused else 'new_fixed_checkpoint_evaluation',**{k:m[k] for k in ('accuracy','macro_f1','macro_precision','macro_recall')}))
        dump(output/'progress.json',results);print('Evaluated',i,'/48',rid,'test_acc',m['accuracy'],'test_F1',m['macro_f1'],flush=True)
    assert all(sha(repo/json.loads(p.read_text())['artifacts']['checkpoint_ref'])==locked[json.loads(p.read_text())['run_id']] for p in records)
    dump(output/'evaluation_report.json',dict(records=48,results=results,checkpoint_hashes_unchanged=True,no_training=True,no_normalization_refit=True,final_model_unchanged='M2_encoder_wavlm_refinement_s42_20261021_05'))
    write_csv(output/'all_test_results.csv',results,['run_id','status','accuracy','macro_f1','macro_precision','macro_recall'])
    print('ALL48_TEST_RECORDS_COMPLETED',flush=True)
if __name__=='__main__':main()
