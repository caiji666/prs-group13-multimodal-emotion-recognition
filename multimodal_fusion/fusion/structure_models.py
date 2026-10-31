"""Exploratory fusion controls; historical models/checkpoints remain unchanged."""
import math
import torch
from torch import nn
from .models import FusionModel, head


METHODS = ('audio_residual', 'audio_residual_mdrop', 'audio_residual_aux',
           'audio_residual_mdrop_aux', 'projected_mean', 'uniform_attention',
           'cross_position', 'cross_temporal', 'cross_bidirectional')


def valid_mask(x, lengths):
    return torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]


def mean_valid(x, lengths):
    return (x * valid_mask(x, lengths)[..., None]).sum(1) / lengths[:, None]


def relative_position(x, lengths):
    """Relative clip index, NOT recovered physical audio/video timestamps."""
    t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[None, :]
    t = t / (lengths - 1).clamp_min(1)[:, None]
    frequencies = torch.arange(1, x.shape[-1] // 2 + 1, device=x.device, dtype=x.dtype)
    angles = 2 * math.pi * t[..., None] * frequencies
    position = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
    return (x + .1 * position) * valid_mask(x, lengths)[..., None]


class StructureModel(nn.Module):
    def __init__(self, method, audio_dim, visual_dim, audio_seq_dim, visual_seq_dim,
                 hidden=64, heads=4, dropout=.2, modality_dropout=.25, aux_weight=.2):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.method = method
        self.modality_dropout = modality_dropout
        self.aux_weight = aux_weight
        config = dict(audio_dim=audio_dim, visual_dim=visual_dim, audio_seq_dim=audio_seq_dim,
                      visual_seq_dim=visual_seq_dim, hidden=hidden, heads=heads, dropout=dropout)
        if method.startswith('audio_residual'):
            self.base = FusionModel('audio_only', **config)
            self.base.requires_grad_(False)
            self.audio = nn.Linear(audio_dim, hidden)
            self.visual = nn.Linear(visual_dim, hidden)
            self.correction = head(2 * hidden, hidden, dropout)
            # Exact audio fallback at initialization; no validation-fitted mixture weight.
            nn.init.zeros_(self.correction[-1].weight)
            nn.init.zeros_(self.correction[-1].bias)
            self.scale = nn.Parameter(torch.tensor(.1))
            self.audio_aux = nn.Linear(hidden, 8)
            self.visual_aux = nn.Linear(hidden, 8)
        else:
            # Identical initialization/parameter budget for uniform vs baseline attention.
            self.core = FusionModel('cross_attention', **config)
            if method == 'projected_mean':
                del self.core.attention
            if method == 'cross_temporal':
                self.audio_conv = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
                self.visual_conv = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
                self.audio_norm = nn.LayerNorm(hidden)
                self.visual_norm = nn.LayerNorm(hidden)
            if method == 'cross_bidirectional':
                self.reverse = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
                self.reverse_norm = nn.LayerNorm(hidden)
                self.joint_classifier = head(2 * hidden, hidden, dropout)
                del self.core.classifier

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'base'):
            self.base.eval()  # Frozen reference includes fixed dropout behavior.
        return self

    def forward(self, batch, return_aux=False):
        if self.method.startswith('audio_residual'):
            self.base.eval()
            a = torch.tanh(self.audio(batch['audio_pooled']))
            v = torch.tanh(self.visual(batch['visual_pooled']))
            clean_v = v
            if self.training and 'mdrop' in self.method:
                keep = (torch.rand(len(v), 1, device=v.device) >= self.modality_dropout)
                v = v * keep  # Train-only missing visual embedding; audio always retained.
            logits = self.base(batch) + self.scale * self.correction(torch.cat((a, v), -1))
            aux = (self.audio_aux(a), self.visual_aux(clean_v)) if 'aux' in self.method else ()
            return (logits, aux) if return_aux else logits

        core = self.core
        a, v = core.audio(batch['audio_sequence']), core.visual(batch['visual_sequence'])
        al, vl = batch['audio_lengths'], batch['visual_lengths']
        am, vm = valid_mask(a, al), valid_mask(v, vl)
        a, v = a * am[..., None], v * vm[..., None]
        if self.method == 'cross_position':
            a, v = relative_position(a, al), relative_position(v, vl)
        if self.method == 'cross_temporal':
            a = self.audio_norm(a + torch.nn.functional.gelu(self.audio_conv(a.transpose(1, 2)).transpose(1, 2))) * am[..., None]
            v = self.visual_norm(v + torch.nn.functional.gelu(self.visual_conv(v.transpose(1, 2)).transpose(1, 2))) * vm[..., None]
        if self.method == 'projected_mean':
            context = mean_valid(v, vl)[:, None, :].expand_as(a)
        elif self.method == 'uniform_attention':
            # Exactly uniform weights over valid visual tokens, with the same learned V/out projections.
            h = a.shape[-1]
            att = core.attention
            value = torch.nn.functional.linear(v, att.in_proj_weight[2*h:], att.in_proj_bias[2*h:])
            context = att.out_proj(mean_valid(value, vl))[:, None, :].expand_as(a)
        else:
            context, _ = core.attention(a, v, v, key_padding_mask=~vm, need_weights=False)
        pooled = mean_valid(core.norm(a + context), al)
        if self.method == 'cross_bidirectional':
            reverse, _ = self.reverse(v, a, a, key_padding_mask=~am, need_weights=False)
            vp = mean_valid(self.reverse_norm(v + reverse), vl)
            logits = self.joint_classifier(torch.cat((pooled, vp), -1))
        else:
            logits = core.classifier(pooled)
        return (logits, ()) if return_aux else logits
