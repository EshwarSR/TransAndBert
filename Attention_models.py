from utils import mask_
import torch
from torch import nn
import torch.nn.functional as F
import random
import math


class SelfAttention(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads

        self.to_keys = nn.Linear(s, s, bias=False)
        self.to_queries = nn.Linear(s, s, bias=False)
        self.to_values = nn.Linear(s, s, bias=False)

        self.project_heads = nn.Linear(heads * s, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads

        s = e // h
        x = x.view(b, t, h, s)

        keys = self.to_keys(x)
        queries = self.to_queries(x)
        values = self.to_values(x)

        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask:
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        out = torch.bmm(dot, values).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.project_heads(out)


class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, seq_length, ff_hidden_mult=4, dropout=0.0):
        super().__init__()

        self.attention = SelfAttention(emb, heads=heads, mask=mask)
        self.mask = mask

        self.layer_norm1 = nn.LayerNorm(emb)
        self.layer_norm2 = nn.LayerNorm(emb)

        self.feed_forward = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        x = self.layer_norm1(attended + x)
        x = self.drop_out(x)
        fedforward = self.feed_forward(x)
        x = self.layer_norm2(fedforward + x)
        x = self.drop_out(x)
        return x


class TransformerClassifier(nn.Module):

    def __init__(self, emb, heads, depth, seq_length, num_tokens, num_classes, max_pool=True, dropout=0.0, device=torch.device("cpu")):
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool

        self.token_embedding = nn.Embedding(
            embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_encoder = nn.Embedding(
            embedding_dim=emb, num_embeddings=seq_length)

        self.device = device

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.drop_out = nn.Dropout(dropout)

    def forward(self, x):
        tokens = self.token_embedding(x)
        b, t, e = tokens.size()

        positional_encodings = self.pos_encoder(torch.arange(t, device=self.device))[
            None, :, :].expand(b, t, e)
        x = tokens + positional_encodings
        x = self.drop_out(x)
        x = self.tblocks(x)
        x = x.max(dim=1)[0] if self.max_pool else x.mean(
            dim=1)  # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)
