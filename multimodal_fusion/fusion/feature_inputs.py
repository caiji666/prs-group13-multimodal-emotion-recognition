"""Feature-only handoff adapter. Test features and upstream probabilities are excluded."""
import csv
import hashlib
import zipfile
from pathlib import Path
import numpy as np
from . import CLASSES
from .data import trial_id


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def selected_npz_rows(path, key, indices):
    """Decode only requested rows of a C-contiguous .npy member, never unpickle.

    Zip seeks may decompress skipped bytes internally, but skipped rows are never
    decoded as arrays or returned to the experiment. No upstream probs are read.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or (np.diff(indices) <= 0).any():
        raise ValueError('Row indices must be nonempty, unique and ascending')
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        if fortran or dtype.hasobject or not shape or indices[0] < 0 or indices[-1] >= shape[0]:
            raise ValueError('Unsupported array format or row indices')
        start = f.tell()
        row_size = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        out = np.empty((len(indices), *shape[1:]), dtype=dtype)
        for j, i in enumerate(indices):
            f.seek(start + int(i) * row_size)
            buf = f.read(row_size)
            if len(buf) != row_size:
                raise ValueError('Truncated array')
            out[j] = np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
    return out


def load_development_features(audio_csv, audio_sequences, visual_npz, actor_csv,
                              expected_counts=None):
    """Only train/val numeric features are decoded; no test file is opened."""
    if expected_counts is None:
        expected_counts = {'train': 960, 'val': 240}
    with Path(actor_csv).open() as f:
        rows = list(csv.DictReader(f))
    actor_map = {int(r['actor_id']): r['split'] for r in rows}
    if len(rows) != 24 or set(actor_map) != set(range(1,25)):
        raise ValueError('Expected 24 unique actors')
    if any(x not in ('train','val','test') for x in actor_map.values()):
        raise ValueError('Unknown actor split')
    # Only IDs and splits are read across the whole visual file to select dev rows.
    with np.load(visual_npz, allow_pickle=False) as z:
        all_ids, all_splits = z['sample_id'], z['split']
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError('Duplicate visual IDs')
    visual_ix = np.flatnonzero(np.isin(all_splits, ['train','val']))
    visual_ids = all_ids[visual_ix]
    visual_splits = all_splits[visual_ix]
    labels = selected_npz_rows(visual_npz, 'label', visual_ix)
    pooled = selected_npz_rows(visual_npz, 'pooled_features', visual_ix)
    sequence = selected_npz_rows(visual_npz, 'sequence_features', visual_ix)
    if pooled.shape != (len(visual_ix),576) or sequence.shape != (len(visual_ix),16,576):
        raise ValueError('Expected MobileNet 576D pooled and 16x576 sequence')
    visual = {}
    for j,(sid,split,label) in enumerate(zip(visual_ids, visual_splits, labels)):
        sid = str(sid)
        if trial_id('03-' + sid + '.wav') != sid:
            raise ValueError('Invalid canonical visual ID')
        if int(label) != int(sid.split('-')[1])-1 or actor_map[int(sid[-2:])] != split:
            raise ValueError('Visual label or split mismatch')
        visual[sid] = (j, str(split), int(label))
    audio = {}
    with Path(audio_csv).open() as f:
        reader = csv.DictReader(f)
        metadata = {'sample_id','trial_id','actor_id','emotion_label','split'}
        feature_keys = [k for k in reader.fieldnames if k not in metadata]
        if len(feature_keys) != 176:
            raise ValueError('Expected 176 combined audio features')
        for row in reader:
            if row['split'] not in ('train','val'):
                continue  # Do not convert/consume test feature values.
            sid = trial_id(row['sample_id'])
            lab = CLASSES.index('surprise' if row['emotion_label']=='surprised' else row['emotion_label'])
            if actor_map[int(row['actor_id'])] != row['split'] or int(sid[-2:]) != int(row['actor_id']):
                raise ValueError('Audio actor/split mismatch')
            if int(sid.split('-')[1])-1 != lab or sid in audio:
                raise ValueError('Audio label mismatch or duplicate ID')
            audio[sid] = (np.asarray([float(row[k]) for k in feature_keys],np.float32),row['split'],lab)
    if set(audio) != set(visual):
        raise ValueError('Development audio/visual sample IDs differ')
    ids = sorted(audio)
    audio_seq, lengths, opened_paths = [], [], []
    for sid in ids:
        ax, split, lab = audio[sid]
        j, vsplit, vlabel = visual[sid]
        if (split,lab) != (vsplit,vlabel):
            raise ValueError('Cross-modal label/split mismatch')
        path = Path(audio_sequences)/split/f'{sid}.npz'
        # Path is derived exclusively from train/val records.
        with np.load(path,allow_pickle=False) as z:
            seq, length, mask = z['F_sequence'], int(z['length']), z['padding_mask']
            if str(z['sample_id']) != sid or str(z['split']) != split or int(z['label']) != lab:
                raise ValueError('Audio sequence metadata mismatch')
            if int(z['actor_id']) != int(sid[-2:]):
                raise ValueError('Audio sequence actor mismatch')
            if seq.shape != (length,128) or length < 1 or mask.dtype != np.bool_ or mask.shape != (length,) or mask.any():
                raise ValueError('Unexpected audio sequence/mask')
            if not np.allclose(seq.mean(0),ax[48:],atol=1e-5):
                raise ValueError('Audio sequence and combined learned feature mismatch')
            audio_seq.append(seq.copy()); lengths.append(length)
        opened_paths.append(path)
    maxlen = max(lengths)
    padded = np.zeros((len(ids),maxlen,128),np.float32)
    for i,s in enumerate(audio_seq):
        padded[i,:len(s)] = s
    order = np.asarray([visual[s][0] for s in ids])
    d = {'sample_ids':np.asarray(ids), 'labels':np.asarray([audio[s][2] for s in ids],np.int64),
         'actors':np.asarray([int(s[-2:]) for s in ids],np.int64),
         'splits':np.asarray([audio[s][1] for s in ids]),
         'audio_pooled':np.stack([audio[s][0] for s in ids]),'visual_pooled':pooled[order],
         'audio_sequence':padded, 'visual_sequence':sequence[order],
         'audio_lengths':np.asarray(lengths,np.int64),'visual_lengths':np.full(len(ids),16,np.int64)}
    for key in ('audio_pooled','visual_pooled','audio_sequence','visual_sequence'):
        if not np.isfinite(d[key]).all():
            raise ValueError('Nonfinite development features: '+key)
    for split,n in expected_counts.items():
        ix = d['splits']==split
        if ix.sum()!=n or set(d['labels'][ix].tolist())!=set(range(8)):
            raise ValueError('Unexpected development counts/classes: '+split)
    source_hashes = {'audio_csv':sha256(audio_csv),'visual_npz':sha256(visual_npz),
                     'actor_csv':sha256(actor_csv)}
    sequence_digest = hashlib.sha256()
    for path in opened_paths:
        sequence_digest.update(path.name.encode())
        sequence_digest.update(bytes.fromhex(sha256(path)))
    source_hashes['ordered_train_val_audio_sequences'] = sequence_digest.hexdigest()
    audit = {'counts':{s:int((d['splits']==s).sum()) for s in ('train','val')},
             'test_status':'not_run', 'test_numeric_rows_decoded':0, 'upstream_probabilities_loaded':False,
             'audio_sequence_files_opened':len(opened_paths),'audio_sequence_splits_opened':['train','val'],
             'source_hashes':source_hashes,
             'notes':['Whole-file IDs/splits used for routing only; only train/val feature rows decoded.',
                      'Shared CSV/NPZ checksums cover whole source containers, not model evaluation.',
                      'Visual 16 tokens treated as valid as supplied; upstream lengths/mask provenance unconfirmed.',
                      'Audio pooled=48 handcrafted+128 learned; attention audio sequence is learned-only 128D.',
                      'Upstream probability order unconfirmed; probability arrays intentionally never loaded.']}
    return d,audit
