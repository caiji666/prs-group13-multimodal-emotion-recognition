import torch
from torch import nn


def head(dim, hidden, dropout):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 8))


class FusionModel(nn.Module):
    def __init__(self, method, audio_dim, visual_dim, audio_seq_dim=1, visual_seq_dim=1,
                 hidden=64, heads=4, dropout=0.2):
        super().__init__()
        self.method = method
        if method in ('audio_only', 'visual_only'):
            self.classifier = head(audio_dim if method == 'audio_only' else visual_dim, hidden, dropout)
        elif method in ('audio_sequence_mean', 'sequence_mean_concat'):
            dim = audio_seq_dim + (visual_seq_dim if method == 'sequence_mean_concat' else 0)
            self.classifier = head(dim, hidden, dropout)
        elif method == 'concat':
            self.classifier = head(audio_dim + visual_dim, hidden, dropout)
        elif method == 'gated':
            self.audio = nn.Linear(audio_dim, hidden)
            self.visual = nn.Linear(visual_dim, hidden)
            self.gate = nn.Linear(audio_dim + visual_dim, hidden)
            self.classifier = head(hidden, hidden, dropout)
        elif method == 'cross_attention':
            if hidden % heads:
                raise ValueError('hidden must be divisible by heads')
            self.audio = nn.Linear(audio_seq_dim, hidden)
            self.visual = nn.Linear(visual_seq_dim, hidden)
            self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.classifier = head(hidden, hidden, dropout)
        else:
            raise ValueError(f'Unknown model {method}')

    @staticmethod
    def sequence_mean(batch, modality):
        x = batch[modality + '_sequence']
        lengths = batch[modality + '_lengths']
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        return (x * valid[..., None]).sum(1) / lengths[:, None]

    def forward(self, batch):
        if self.method in ('audio_only', 'visual_only'):
            modality = self.method.split('_')[0]
            return self.classifier(batch[modality + '_pooled'])
        if self.method in ('audio_sequence_mean', 'sequence_mean_concat'):
            pooled = self.sequence_mean(batch, 'audio')
            if self.method == 'sequence_mean_concat':
                pooled = torch.cat([pooled, self.sequence_mean(batch, 'visual')], -1)
            return self.classifier(pooled)
        if self.method == 'concat':
            return self.classifier(torch.cat([batch['audio_pooled'], batch['visual_pooled']], -1))
        if self.method == 'gated':
            a, v = batch['audio_pooled'], batch['visual_pooled']
            g = torch.sigmoid(self.gate(torch.cat([a, v], -1)))
            fused = g * torch.tanh(self.audio(a)) + (1 - g) * torch.tanh(self.visual(v))
            return self.classifier(fused)
        a, v = self.audio(batch['audio_sequence']), self.visual(batch['visual_sequence'])
        mask_v = torch.arange(v.shape[1], device=v.device)[None, :] >= batch['visual_lengths'][:, None]
        mask_a = torch.arange(a.shape[1], device=a.device)[None, :] < batch['audio_lengths'][:, None]
        context, _ = self.attention(a, v, v, key_padding_mask=mask_v, need_weights=False)
        sequence = self.norm(a + context)
        pooled = (sequence * mask_a[..., None]).sum(1) / batch['audio_lengths'][:, None]
        return self.classifier(pooled)
