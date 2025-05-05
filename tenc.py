import math
import torch
from torch import nn
from modules import *  # MultiHeadAttention, PositionwiseFeedForward
from utility import extract_axis_1


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Tenc(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, diffuser_type, device, num_heads=1):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.device = device
        # embeddings
        self.item_embeddings = nn.Embedding(item_num+1, hidden_size)
        nn.init.normal_(self.item_embeddings.weight, 0, 1)
        self.none_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        self.positional_embeddings = nn.Embedding(state_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        # transformer
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(
            hidden_size, hidden_size, num_heads, dropout)
        self.ff = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        # step embedding
        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_size),
            nn.Linear(hidden_size, hidden_size*2),
            nn.GELU(),
            nn.Linear(hidden_size*2, hidden_size)
        )
        # diffuser
        if diffuser_type == 'mlp1':
            self.diffuser = nn.Linear(hidden_size*3, hidden_size)
        else:
            self.diffuser = nn.Sequential(
                nn.Linear(hidden_size*3, hidden_size*2),
                nn.GELU(),
                nn.Linear(hidden_size*2, hidden_size)
            )

    def forward(self, x, h, step):
        t = self.step_mlp(step)
        return self.diffuser(torch.cat((x, h, t), dim=1))

    def forward_uncon(self, x, step):
        B = x.size(0)
        h0 = self.none_embedding(torch.zeros(
            1, device=self.device, dtype=torch.long)).expand(B, -1)
        emb = self.step_mlp(step.expand(B))
        return self.diffuser(torch.cat((x, h0, emb), dim=1))

    def cacu_x(self, x):
        return self.item_embeddings(x)

    def cacu_h(self, states, len_states, p):
        emb = self.item_embeddings(states)
        emb = emb + \
            self.positional_embeddings(torch.arange(
                self.state_size, device=self.device))
        seq = self.emb_dropout(emb)
        mask = (states != self.item_num).float().unsqueeze(-1).to(self.device)
        seq = seq * mask
        x = self.ln1(seq)
        x = self.mh_attn(x, seq)
        x = self.ff(self.ln2(x)) * mask
        x = self.ln3(x)
        last = extract_axis_1(x, len_states-1)
        h = last.squeeze()
        # dropout in hidden
        drop = (torch.rand(h.size(0), device=self.device)
                > p).float().unsqueeze(1)
        h = h*drop + self.none_embedding(torch.zeros(
            1, dtype=torch.long, device=self.device)).expand_as(h)*(1-drop)
        return h

    def predict(self, states, len_states, diff, steps=1):
        h = self.cacu_h(states, len_states, p=0.0)
        # one-step sampling
        x0 = diff.sample(self.forward, self.forward_uncon,
                         h, self.device, steps)
        emb = self.item_embeddings.weight  # (item_num+1, hidden_size)
        scores = torch.matmul(x0, emb.T)
        return scores
