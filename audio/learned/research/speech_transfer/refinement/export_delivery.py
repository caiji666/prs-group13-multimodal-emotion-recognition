"""Export the frozen WavLM winner using the legacy M5 directory and field contract."""
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from .inference import SpeechTransferInference
from ..prepare import ROOT,sha,REVISION
from ravdess_m2.data import manifest,LABELS

AUDIO='F_audio_learned_feature_only'
PROB='P_audio_learned_feature_only'
def write_csv(path,rows,fields):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default=str(ROOT/'runs/refinement_20261021/round_05_temporal_context/best.pt'))
    p.add_argument('--output',default=str(ROOT.parents[3]/'Deliverables/wavlm_refinement05_new'))
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    target=out/'m5_learned_feature_only';target.mkdir();(target/'pooled').mkdir()
    split_file=ROOT.parents[3]/'multimodal_fusion/configs/actor_splits.csv'
    rows=manifest(ROOT.parents[1]/'data/raw',split_file)
    assert len(rows)==1440 and len({r['sample_id'] for r in rows})==1440
    assert Counter(r['split'] for r in rows)==dict(train=960,val=240,test=240)
    engine=SpeechTransferInference(a.checkpoint)
    assert engine.saved['split_sha256']==sha(split_file)
    index=[];groups={s:[] for s in ('train','val','test')};hashes={};start=time.monotonic()
    for i,row in enumerate(rows):
        result=engine.predict(row['path']);sid=row['sample_id'];split=row['split'];n=int(result['length'])
        for k in ('sample_id','actor_id','split','label'):assert result[k]==row[k]
        assert result['F_sequence'].shape==(n,128) and result[AUDIO].shape==(128,) and result[PROB].shape==(8,)
        assert all(np.isfinite(result[k]).all() for k in ('F_sequence',AUDIO,PROB))
        assert np.isclose(result[PROB].sum(),1,atol=1e-6)
        dest=target/'per_sample'/split/(sid+'.npz');dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,sample_id=sid,actor_id=np.int64(row['actor_id']),split=split,label=np.int64(row['label']),
            F_sequence=result['F_sequence'],length=np.int64(n),padding_mask=result['padding_mask'],class_names=np.array(LABELS),
            representation_role='M2_learned_feature_only_for_M5_testing',**{AUDIO:result[AUDIO],PROB:result[PROB]})
        with np.load(dest,allow_pickle=False) as z:
            for k in ('F_sequence',AUDIO,PROB):np.testing.assert_array_equal(z[k],result[k])
            assert z['padding_mask'].shape==(n,) and not z['padding_mask'].any()
        rel=dest.relative_to(target).as_posix();hashes[rel]=sha(dest)
        index.append(dict(sample_id=sid,actor_id=row['actor_id'],split=split,label=row['label'],length=n,feature_file=rel))
        groups[split].append((row,result[AUDIO],result[PROB]))
        if (i+1)%50==0:print(f'Exported {i+1}/1440; elapsed {time.monotonic()-start:.1f}s',flush=True)
    for split,items in groups.items():
        pooled=np.stack([x[1] for x in items]);probs=np.stack([x[2] for x in items])
        np.savez_compressed(target/'pooled'/f'{split}.npz',sample_ids=np.array([x[0]['sample_id'] for x in items]),
            labels=np.array([x[0]['label'] for x in items]),actor_ids=np.array([x[0]['actor_id'] for x in items]),
            splits=np.array([split]*len(items)),class_names=np.array(LABELS),**{AUDIO:pooled,PROB:probs})
        fields=['sample_id','actor_id','split','label']+[f'{AUDIO}_{j:03d}' for j in range(128)]+[f'{PROB}_{c}' for c in LABELS]
        records=[]
        for row,pool,prob in items:
            records.append(dict(zip(fields,[row['sample_id'],row['actor_id'],split,row['label'],*pool.tolist(),*prob.tolist()])))
        write_csv(target/'pooled'/f'{split}.csv',records,fields)
    write_csv(target/'manifest.csv',index,list(index[0]))
    metadata=dict(samples=1440,split_counts=dict(Counter(r['split'] for r in rows)),class_names=LABELS,
        fields=dict(F_sequence='[T,128]',F_audio_learned_feature_only='[128]',P_audio_learned_feature_only='[8]'),
        model='frozen WavLM Base+ layer 6 + temporal adapter + residual depthwise temporal convolution kernel3 + attentive statistics pooling + MLP',
        checkpoint=str(Path(a.checkpoint).resolve()),selection_run_id='M2_encoder_wavlm_refinement_s42_20261021_05',validation_accuracy=0.8416666666666667,validation_macro_f1=0.8312181237730203,checkpoint_sha256=sha(a.checkpoint),upstream_revision=REVISION,
        upstream_sha256=engine.saved['upstream']['sha256'],split_sha256=sha(split_file),
        handcrafted_fusion=False,final_M1_audio_outputs=False,padding_mask='True = padding; stored per-sample arrays are unpadded',
        pooling='learned attention weighted mean/std followed by learned projection; NOT arithmetic mean of F_sequence',
        sequence_timing='16k waveform after boundary silence trim; WavLM time steps, not legacy MFCC time steps; audio/video alignment requires accounting for trimming',
        test_audio_decoded=True,test_metrics_computed=False,model_selection_performed=False,augmentation=False,
        training_probabilities='in-sample; not out-of-fold stacking predictions',elapsed_seconds=time.monotonic()-start)
    (target/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (target/'checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (target/'README.md').write_text('''# WavLM continuation 20261021 Round05 (residual temporal convolution kernel3) audio handoff for M5\n\nThe directory/NPZ field contract follows the previous M2 learned-only delivery. Each per_sample/{train,val,test}/{sample_id}.npz contains F_sequence [T,128], F_audio_learned_feature_only [128], P_audio_learned_feature_only [8], sample_id, actor_id, split, label, length, padding_mask and class_names. Load with numpy.load(path, allow_pickle=False). manifest.csv indexes every sample. pooled/{split}.npz and CSV contain aligned pooled/probability tables.\n\nF_sequence is M2's trained temporal adapter sequence for cross-attention. F_audio_learned_feature_only is the 128-dimensional attentive-statistics pooled representation, without M1 handcrafted features. It is NOT sequence.mean(0). P_audio_learned_feature_only is softmax of this M2 model's MLP; it is NOT M1's final audio classifier probability.\n\nMatch six-field sample_id, label, split and class order before fusion. Class order: neutral, calm, happy, sad, angry, fearful, disgust, surprise (labels 0..7). Pad variable sequences when batching; True indicates padding, and stored individual masks are all False. WavLM sequence time steps differ from MFCC, and boundary silence trimming must be considered for precise audio/video alignment.\n\nThe backbone and selected Round 05 adapter were fixed before export; no augmentation, training, model selection or test metrics computation occurred during export. Test WAVs were decoded solely to generate the held-out fusion inputs. Train fusion on train, select with val and reserve test for final evaluation. Train probabilities are in-sample, not out-of-fold. See metadata.json for provenance; checksums.json lists per-sample SHA-256.\n''')
    (out/'README.md').write_text('WavLM continuation 20261021 Round05 (residual temporal convolution kernel3) delivery: send the complete m5_learned_feature_only folder to M5. See its README and metadata. Best validation Accuracy 0.8416666666666667 / Macro-F1 0.8312181237730203; no test metrics computed.\n')
    print('Completed:',target,flush=True)
if __name__=='__main__':main()
