import copy
import csv
import json
import platform
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from . import CLASSES
from .data import load_pair, split_indices, fit_normalizers, normalize
from .models import FusionModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def tensor_batch(d, ix, device):
    return {k: torch.as_tensor(x[ix], device=device) for k,x in d.items()
            if k.startswith(('audio_', 'visual_')) and not k.endswith('_probs')}


@torch.no_grad()
def infer(model, d, ix, device, batch_size=64):
    model.eval()
    return np.concatenate([model(tensor_batch(d, ix[j:j+batch_size], device)).softmax(-1).cpu().numpy()
                           for j in range(0, len(ix), batch_size)])


def metrics(y, probs):
    pred = probs.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(8), average='macro', zero_division=0)),
            'per_class': classification_report(y, pred, labels=range(8), target_names=CLASSES,
                                               output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(8)).tolist()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def run(audio, visual, split_path, output, methods, epochs=30, batch_size=64, lr=.001,
        hidden=64, heads=4, dropout=.2, patience=8, seed=42, device='cpu', synthetic=False):
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0:
        raise ValueError('epochs, batch_size, patience and lr must be positive')
    output = Path(output)
    if (output / 'metrics.json').exists():
        raise ValueError('Output already contains a run; choose a new --output to preserve results')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
    seed_all(seed)
    raw = load_pair(audio, visual)
    synthetic = synthetic or raw.get('data_kind') == 'synthetic'
    indices, actors = split_indices(raw, split_path)
    if 'cross_attention' in methods and 'audio_sequence' not in raw:
        raise ValueError('cross_attention requires audio sequences')
    if 'cross_attention' in methods and 'visual_sequence' not in raw:
        raise ValueError('cross_attention requires visual sequences')
    train, val, test = (indices[k] for k in ('train', 'val', 'test'))
    normalizers = fit_normalizers(raw, train)
    d = normalize(raw, normalizers)
    config = dict(audio_dim=raw['audio_pooled'].shape[1], visual_dim=raw['visual_pooled'].shape[1],
                  audio_seq_dim=raw.get('audio_sequence', np.empty((0,0,1))).shape[-1],
                  visual_seq_dim=raw.get('visual_sequence', np.empty((0,0,1))).shape[-1],
                  hidden=hidden, heads=heads, dropout=dropout)
    report = {'synthetic': synthetic, 'seed': seed, 'actors': actors,
              'counts': {k: len(v) for k,v in indices.items()}, 'class_names': list(CLASSES),
              'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__)},
              'settings': dict(epochs=epochs, batch_size=batch_size, lr=lr, patience=patience, device=device, **config),
              'inputs': {'audio': str(Path(audio).resolve()), 'visual': str(Path(visual).resolve())},
              'models': {}}
    predictions = {}
    for name, key in [('audio_only', 'audio_probs'), ('visual_only', 'visual_probs')]:
        report['models'][name] = {part: metrics(d['labels'][ix], d[key][ix]) for part, ix in [('val', val), ('test', test)]}
        predictions[name] = d[key][test]
    if 'decision' in methods:
        def mix(alpha, ix):
            return alpha * d['audio_probs'][ix] + (1-alpha) * d['visual_probs'][ix]
        grid = [dict(alpha=float(a), macro_f1=metrics(d['labels'][val], mix(a,val))['macro_f1'])
                for a in np.linspace(0,1,21)]
        # Ties favor the least departure from equal weighting.
        winner = max(grid, key=lambda row: (row['macro_f1'], -abs(row['alpha']-.5)))
        for name, alpha in [('decision_equal', .5), ('decision_tuned', winner['alpha'])]:
            report['models'][name] = {'alpha_audio': alpha, 'val': metrics(d['labels'][val], mix(alpha,val)),
                                      'test': metrics(d['labels'][test], mix(alpha,test))}
            predictions[name] = mix(alpha,test)
            write_json(output / 'checkpoints' / f'{name}.json', {'method':'decision', 'alpha_audio': alpha, 'class_names': list(CLASSES)})
        write_json(output / 'decision_validation_grid.json', grid)
    histories = {}
    for method in methods:
        if method == 'decision':
            continue
        seed_all(seed)
        model = FusionModel(method, **config).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best, best_epoch, stale, best_state = -1, 0, 0, None
        history = []
        for epoch in range(1, epochs+1):
            model.train()
            order = np.random.permutation(train)
            loss_sum = 0.
            for j in range(0, len(order), batch_size):
                ix = order[j:j+batch_size]
                opt.zero_grad()
                logits = model(tensor_batch(d, ix, device))
                loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(d['labels'][ix], device=device))
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss in {method}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                loss_sum += loss.item() * len(ix)
            score = metrics(d['labels'][val], infer(model,d,val,device,batch_size))['macro_f1']
            history.append({'epoch':epoch, 'train_loss':loss_sum/len(train), 'val_macro_f1':score})
            if score > best + 1e-8:
                best, best_epoch, stale = score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= patience:
                break
        model.load_state_dict(best_state)
        checkpoint = {'method':method, 'config':config, 'state_dict':best_state,
                      'normalizers':normalizers, 'class_names':list(CLASSES), 'actors':actors,
                      'synthetic':synthetic, 'best_epoch':best_epoch}
        torch.save(checkpoint, output / 'checkpoints' / f'{method}.pt')
        report['models'][method] = {'best_epoch':best_epoch,
                                   'val':metrics(d['labels'][val], infer(model,d,val,device,batch_size)),
                                   'test':metrics(d['labels'][test], infer(model,d,test,device,batch_size))}
        predictions[method] = infer(model,d,test,device,batch_size)
        histories[method] = history
        print(f'{method}: best epoch={best_epoch}, validation macro-F1={best:.4f}', flush=True)
    # Choose by validation only. Test scores are descriptive comparisons, not selection inputs.
    selected = max(report['models'], key=lambda k:report['models'][k]['val']['macro_f1'])
    baseline = max(('audio_only','visual_only'), key=lambda k:report['models'][k]['val']['macro_f1'])
    report['selected_by_validation'] = selected
    report['strongest_unimodal_by_validation'] = baseline
    report['selected_test_delta_vs_baseline'] = report['models'][selected]['test']['macro_f1'] - report['models'][baseline]['test']['macro_f1']
    write_json(output / 'metrics.json', report)
    write_json(output / 'history.json', histories)
    with (output / 'summary.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','val_macro_f1','test_macro_f1','test_accuracy'])
        for name,row in report['models'].items():
            writer.writerow([name, row['val']['macro_f1'], row['test']['macro_f1'], row['test']['accuracy']])
    with (output / 'test_predictions.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','label','model','prediction',*CLASSES])
        for name,probs in predictions.items():
            for sid,label,p in zip(d['sample_ids'][test], d['labels'][test],probs):
                writer.writerow([sid, int(label), name, int(p.argmax()), *p.tolist()])
    # Inspectable complete split assignment, shared with teammates.
    with (output / 'split_manifest.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_id','actor','label','split'])
        for part,ix in indices.items():
            writer.writerows(zip(d['sample_ids'][ix], d['actors'][ix], d['labels'][ix], [part]*len(ix)))
    render_confusions(report, output)
    return report


def render_confusions(report, output):
    import os
    import tempfile
    cache = Path(tempfile.gettempdir()) / 'multimodal_fusion_matplotlib'
    cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,row in report['models'].items():
        cm = np.array(row['test']['confusion_matrix'])
        fig,ax = plt.subplots(figsize=(7,6))
        ax.imshow(cm, cmap='Blues')
        for i in range(8):
            for j in range(8):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j] > cm.max()/2 else 'black')
        ax.set(xticks=range(8), yticks=range(8), xticklabels=CLASSES, yticklabels=CLASSES,
               xlabel='Predicted', ylabel='True', title=('SYNTHETIC TEST — ' if report['synthetic'] else 'Test — ') + name)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(output / f'confusion_{name}.png',dpi=130)
        plt.close(fig)


def predict(checkpoint_path, audio, visual, output, device='cpu'):
    d = load_pair(audio,visual)
    if str(checkpoint_path).endswith('.json'):
        ck = json.loads(Path(checkpoint_path).read_text())
        if tuple(ck['class_names']) != CLASSES:
            raise ValueError('Checkpoint class order mismatch')
        alpha = ck['alpha_audio']
        probs = alpha*d['audio_probs'] + (1-alpha)*d['visual_probs']
    else:
        ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if tuple(ck['class_names']) != CLASSES:
            raise ValueError('Checkpoint class order mismatch')
        model = FusionModel(ck['method'], **ck['config']).to(device)
        model.load_state_dict(ck['state_dict'])
        d = normalize(d,ck['normalizers'])
        probs = infer(model,d,np.arange(len(d['labels'])),device)
    rows = [{'sample_id':str(sid), 'emotion':CLASSES[int(p.argmax())],
             'confidence':float(p.max()), 'probabilities':dict(zip(CLASSES,map(float,p)))}
            for sid,p in zip(d['sample_ids'],probs)]
    write_json(output,rows)
    return probs
