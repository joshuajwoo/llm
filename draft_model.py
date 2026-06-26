"""
Draft Model — 1-layer, 64-dim transformer for speculative decoding proposals.
~37K params vs ~10.8M target. Trained 500 steps, just enough to correlate with target.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

import gpt

draft_n_embd = 64
draft_train_steps = 500


class DraftModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(gpt.vocab_size, draft_n_embd)   # (65, 64)
        self.position_embedding_table = nn.Embedding(gpt.block_size, draft_n_embd) # (256, 64)

        # Single attention head — no multi-head split
        self.key = nn.Linear(draft_n_embd, draft_n_embd, bias=False)    # (64, 64)
        self.query = nn.Linear(draft_n_embd, draft_n_embd, bias=False)  # (64, 64)
        self.value = nn.Linear(draft_n_embd, draft_n_embd, bias=False)  # (64, 64)
        self.register_buffer('tril', torch.tril(torch.ones(gpt.block_size, gpt.block_size)))

        self.lm_head = nn.Linear(draft_n_embd, gpt.vocab_size)  # (64, 65)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                          # (B, T, 64)
        pos_emb = self.position_embedding_table(torch.arange(T, device=gpt.device))  # (T, 64)
        x = tok_emb + pos_emb                                             # (B, T, 64)

        # Scaled dot-product self-attention (single head)
        k = self.key(x)                                                    # (B, T, 64)
        q = self.query(x)                                                  # (B, T, 64)
        wei = q @ k.transpose(-2, -1) * (draft_n_embd**-0.5)              # (B, T, T) — scaled scores
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))       # causal mask
        wei = F.softmax(wei, dim=-1)                                       # (B, T, T) — attention weights
        v = self.value(x)                                                  # (B, T, 64)
        x = wei @ v                                                        # (B, T, 64)

        logits = self.lm_head(x)                                           # (B, T, 65)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
