"""Versioned final-delivery adapter; explicit split reads and no mean-pool assumption for WavLM."""
import csv
import json
from pathlib import Path
import numpy as np
from . import CLASSES
from .data import trial_id
from .feature_inputs import selected_npz_rows,sha256


def classes(values):
    return tuple('surprise' if str(c)=='surprised' else str(c) for c in values)


def check_prob(p):
    if p.ndim!=2 or p.shape[1]!=8 or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Invalid class probabilities')


class FinalInputs:
    def __init__(self,audio,visual,actors,handcrafted):
        self.audio=Path(audio);self.visual=Path(visual);self.handcrafted=Path(handcrafted)
        self.meta=json.loads((self.audio/'metadata.json').read_text())
        self.checksums=json.loads((self.audio/'checksums.json').read_text())
        with Path(actors).open() as f:self.actors={int(r['actor_id']):r['split'] for r in csv.DictReader(f)}
        assert len(self.actors)==24 and self.meta['split_sha256']==sha256(actors)
        assert classes(self.meta['class_names'])==CLASSES
        with np.load(self.visual,allow_pickle=False) as z:
            self.ids=z['sample_id'].astype(str);self.splits=z['split'].astype(str)
            assert classes(z['class_names'])==CLASSES
            self.visual_metadata={k:z[k].tolist() for k in ('selected_representation','selected_classifier','selection_metric',
                  'selection_split','test_used_for_selection','deep_source_model','deep_source_checkpoint','train_probability_protocol')}
        assert len(self.ids)==len(set(self.ids))==1440
        assert self.visual_metadata['selected_representation']=='A' and self.visual_metadata['test_used_for_selection'] is False
        for sid,split in zip(self.ids,self.splits):
            assert trial_id('03-'+sid+'.wav')==sid and self.actors[int(sid[-2:])]==split
        self.hashes=dict(audio_metadata=sha256(self.audio/'metadata.json'),audio_checksums=sha256(self.audio/'checksums.json'),
                         visual_npz=sha256(self.visual),actor_csv=sha256(actors),audio_handcrafted=sha256(self.handcrafted))

    def load(self,parts):
        parts=tuple(parts);assert set(parts)<= {'train','val','test'}
        indices=np.flatnonzero(np.isin(self.splits,parts));ids=sorted(self.ids[indices].tolist())
        lookup={sid:int(i) for i,sid in enumerate(self.ids)};order=np.array([list(indices).index(lookup[sid]) for sid in ids])
        get=lambda key:selected_npz_rows(self.visual,key,indices)[order]
        labels=get('label');actors=get('actor_id');splits=np.array([self.splits[lookup[sid]] for sid in ids])
        assert np.array_equal(labels,np.array([int(s.split('-')[1])-1 for s in ids]))
        assert np.array_equal(actors,np.array([int(s[-2:]) for s in ids]))
        result=dict(sample_ids=np.array(ids),labels=labels.astype(np.int64),actors=actors.astype(np.int64),splits=splits)
        sequences=[];pooled=[];probs=[];file_hashes=[];audio_tables={}
        for split in parts:
            with np.load(self.audio/'pooled'/f'{split}.npz',allow_pickle=False) as z:
                assert classes(z['class_names'])==CLASSES
                table={str(s):j for j,s in enumerate(z['sample_ids'])};assert len(table)==len(z['sample_ids'])
                audio_tables[split]=dict(ids=table,pooled=z['F_audio_learned_feature_only'],probs=z['P_audio_learned_feature_only'],
                     labels=z['labels'],actors=z['actor_ids'],splits=z['splits'])
        for sid,lab,actor,split in zip(ids,labels,actors,splits):
            rel=f'per_sample/{split}/{sid}.npz';path=self.audio/rel;digest=sha256(path)
            assert digest==self.checksums[rel];file_hashes.append((rel,digest))
            with np.load(path,allow_pickle=False) as z:
                assert str(z['sample_id'])==sid and int(z['label'])==lab and int(z['actor_id'])==actor and str(z['split'])==split
                assert classes(z['class_names'])==CLASSES
                seq=z['F_sequence'];length=int(z['length']);mask=z['padding_mask'];a=z['F_audio_learned_feature_only'];p=z['P_audio_learned_feature_only']
                assert seq.shape==(length,128) and length>0 and mask.shape==(length,) and mask.dtype==np.bool_ and not mask.any()
                assert a.shape==(128,) and p.shape==(8,) and np.isfinite(seq).all() and np.isfinite(a).all()
                t=audio_tables[str(split)];j=t['ids'][sid]
                assert int(t['labels'][j])==lab and int(t['actors'][j])==actor and str(t['splits'][j])==split
                np.testing.assert_array_equal(a,t['pooled'][j]);np.testing.assert_array_equal(p,t['probs'][j])
                sequences.append(seq);pooled.append(a);probs.append(p)
        lengths=np.array([len(s) for s in sequences],np.int64)
        padded=np.zeros((len(ids),lengths.max(),128),np.float32)
        for i,seq in enumerate(sequences):padded[i,:len(seq)]=seq
        result.update(audio_pooled=np.stack(pooled),audio_sequence=padded,audio_lengths=lengths,audio_probs=np.stack(probs))
        check_prob(result['audio_probs'])
        for name,field,pfield,seqfield,dim in [('hc','F_visual_handcrafted','P_visual_handcrafted_fusion','F_visual_handcrafted_sequence',320),
                       ('deep','F_visual_deep','P_visual_deep_mlp_fusion','F_visual_deep_sequence',576)]:
            x=get(field);s=get(seqfield);p=get(pfield)
            assert x.shape==(len(ids),dim) and s.shape==(len(ids),16,dim)
            assert np.isfinite(x).all() and np.isfinite(s).all();check_prob(p)
            np.testing.assert_allclose(x,s.mean(1),atol=3e-6,rtol=1e-5)
            result.update({f'visual_{name}_pooled':x,f'visual_{name}_sequence':s,f'visual_{name}_probs':p})
        result['visual_combined_pooled']=get('F_visual_combined')
        np.testing.assert_allclose(result['visual_combined_pooled'],np.concatenate((result['visual_hc_pooled'],result['visual_deep_pooled']),1),atol=1e-6)
        result['visual_combined_sequence']=np.concatenate((result['visual_hc_sequence'],result['visual_deep_sequence']),-1)
        result['visual_combined_probs']=get('P_visual_combined_fusion');check_prob(result['visual_combined_probs'])
        np.testing.assert_array_equal(get('F_visual'),result['visual_hc_pooled'])
        np.testing.assert_array_equal(get('P_visual_fusion'),result['visual_hc_probs'])
        result['visual_lengths']=np.full(len(ids),16,np.int64)
        frames=get('frame_idx');times=get('frame_time_sec');fps=get('video_fps')
        assert (fps>0).all() and (np.diff(frames,axis=1)>=0).all()
        np.testing.assert_allclose(times,frames/fps[:,None],atol=.002,rtol=1e-4)
        with self.handcrafted.open() as f:
            reader=csv.DictReader(f);keys=[k for k in reader.fieldnames if k not in ('sample_id','trial_id','actor_id','emotion_label','split')]
            assert len(keys)==48;hc={}
            for row in reader:
                if row['split'] not in parts:continue
                sid=trial_id(row['sample_id']);assert sid not in hc and sid in ids
                assert int(row['actor_id'])==int(sid[-2:]) and self.actors[int(row['actor_id'])]==row['split']
                label='surprise' if row['emotion_label']=='surprised' else row['emotion_label']
                assert CLASSES.index(label)==int(sid.split('-')[1])-1
                hc[sid]=[float(row[k]) for k in keys]
        assert set(hc)==set(ids);result['audio_handcrafted_pooled']=np.asarray([hc[s] for s in ids],np.float32)
        assert np.isfinite(result['audio_handcrafted_pooled']).all()
        result['audio_combined_pooled']=np.concatenate((result['audio_handcrafted_pooled'],result['audio_pooled']),1)
        audit=dict(parts=list(parts),counts={s:int((splits==s).sum()) for s in parts},source_hashes=self.hashes,
           audio_metadata=self.meta,visual_metadata=self.visual_metadata,audio_file_checksums_verified=len(ids),
           visual_assumptions='16 valid frames supplied; handcrafted sequence mean exactly pooled, deep mean checked; combined sequence concatenated from the aligned supplied sequences.',
           audio_pooling='Supplied attentive statistics vector; no sequence-mean equality required.',
           audio_train_probabilities='in-sample; no train-time probability stacking used',
           sequence_alignment='Relative token positions only; physical audio trim offset not provided.',test_numeric_rows_loaded=int((splits=='test').sum()))
        return result,audit


def view(raw,visual='hc',audio='learned'):
    d={k:raw[k] for k in ('sample_ids','labels','actors','splits','audio_sequence','audio_lengths','visual_lengths')}
    d['audio_pooled']=raw['audio_pooled'] if audio=='learned' else raw[f'audio_{audio}_pooled']
    d['visual_pooled']=raw[f'visual_{visual}_pooled'];d['visual_sequence']=raw[f'visual_{visual}_sequence']
    return d
