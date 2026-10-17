import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AudioEncoder(nn.Module):
    def __init__(self, kind, input_dim=40, hidden=128):
        super().__init__()
        if kind not in ('cnn','bilstm','transformer'):
            raise ValueError(kind)
        self.kind = kind
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        if kind == 'cnn':
            self.conv1 = nn.Conv1d(input_dim,hidden,3,padding=1)
            self.conv2 = nn.Conv1d(hidden,hidden,3,padding=1)
        elif kind == 'bilstm':
            self.encoder = nn.LSTM(input_dim,hidden//2,batch_first=True,bidirectional=True)
        else:
            self.proj = nn.Linear(input_dim,hidden)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=.1,batch_first=True),2,enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(hidden,64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64,8))
    def forward(self, x, lengths):
        lengths = lengths.to(x.device)
        if x.ndim != 3 or lengths.shape != (x.shape[0],) or (lengths < 1).any() or (lengths > x.shape[1]).any():
            raise ValueError('Expected [B,T,D] and positive valid lengths')
        mask = torch.arange(x.shape[1],device=x.device)[None,:] >= lengths[:,None]
        x = ((x-self.feature_mean)/self.feature_std).masked_fill(mask[:,:,None],0)
        if self.kind == 'cnn':
            x = torch.relu(self.conv1(x.transpose(1,2))).masked_fill(mask[:,None,:],0)
            x = torch.relu(self.conv2(x)).masked_fill(mask[:,None,:],float('-inf'))
            x = nn.functional.max_pool1d(x,2,2,ceil_mode=True).transpose(1,2)
            lengths = (lengths+1)//2
        elif self.kind == 'bilstm':
            x,_ = self.encoder(pack_padded_sequence(x,lengths.cpu(),batch_first=True,enforce_sorted=False))
            x,_ = pad_packed_sequence(x,batch_first=True,total_length=mask.shape[1])
        else:
            x = self.proj(x)
            pos = torch.arange(x.shape[1],device=x.device)[:,None]
            div = torch.exp(torch.arange(0,x.shape[2],2,device=x.device)*(-math.log(10000)/x.shape[2]))
            pe = torch.zeros_like(x[0]); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
            x = self.encoder(x+pe,src_key_padding_mask=mask)
        mask = torch.arange(x.shape[1],device=x.device)[None,:] >= lengths[:,None]
        sequence = x.masked_fill(mask[:,:,None],0)
        pooled = sequence.sum(1)/lengths[:,None]
        return dict(sequence=sequence, pooled=pooled, lengths=lengths, padding_mask=mask, logits=self.head(pooled))
