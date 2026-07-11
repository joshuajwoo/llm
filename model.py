"""
A custom decoder architecture combining:
- Differential Attention (Microsoft DIFF Transformer, 2024): two attention maps
  subtracted to cancel noise, with learnable lambda reparameterization
- Adaptive-depth early exit: tokens can exit at trained checkpoints between
  stages when the model is confident it already knows the answer
- Multi-token prediction (MTP) at each exit: auxiliary t+2 loss gives early
  layers a stronger, more direct gradient signal during training
- Hybrid local/global attention: local layers use sliding window + learned
  sink tokens, global layers (last in each stage) use full causal attention
- Standard modern backbone: RoPE, RMSNorm (pre-norm), SwiGLU FFN, weight tying

No single existing model uses this combination.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import ModelConfig


# Faster than LayerNorm, skips mean-centering
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x * 1/sqrt(x^2/n + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

# Rotary Position Embeddings (RoPE) — no learned params, length-generalizable
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """Precompute complex exponentials for rotary embeddings.

    dim: sub_head_dim = head_dim // 2 (because differential attention
         splits Q/K into two sub-heads of half the size)
    max_seq_len: block_size + n_sinks
    Returns: Shape 
        (max_seq_len, dim // 2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len) # 0, 1, 2, ... max_seq_len
    freqs = torch.outer(t, freqs)  # (max_seq_len, dim // 2)
    return torch.polar(torch.ones_like(freqs), freqs) # (r * e^i*theta)

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to query and key sub-heads.

    Args: 
        xq: (B, n_head, T, sub_head_dim)
        xk: (B, n_head, T, sub_head_dim)
        freqs_cis: (T, sub_head_dim // 2) complex tensor
    Returns: 
        Rotated (xq, xk) with same shapes
    """
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # xq shape -> (B, n_head, T, sub_head_dim//2, 2)
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) # xk shape -> (B, n_head, T, sub_head_dim//2, 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)  # (1, 1, T, dim // 2) prepare for broadcasting
    xq_out = torch.view_as_real(xq_c * freqs_cis).flatten(-2) # (B, n_head, T, sub_head_dim)
    xk_out = torch.view_as_real(xk_c * freqs_cis).flatten(-2) # (B, n_head, T, sub_head_dim)
    return xq_out.type_as(xq), xk_out.type_as(xk) # match type to arguments

# KV Cache — O(1) per-token generation, with rollback for speculative decoding
class KVCache:
    """Fixed-size cache for autoregressive generation.

    Stores K and V for all previously-seen positions.  rollback(pos)
    truncates the cache for speculative-decoding rejection.
    """

    def __init__(self, batch_size, max_seq_len, n_heads, head_dim, device):
        self.cache_k = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim, device=device
        )
        self.cache_v = torch.zeros(
            batch_size, n_heads, max_seq_len, head_dim, device=device
        )
        self.seq_len = 0

    def update(self, k, v): 
        """Append new K, V to cache and return the full cached sequence."""
        T_new = k.shape[2]
        self.cache_k[:, :, self.seq_len : self.seq_len + T_new, :] = k
        self.cache_v[:, :, self.seq_len : self.seq_len + T_new, :] = v
        self.seq_len += T_new
        return (
            self.cache_k[:, :, : self.seq_len, :],
            self.cache_v[:, :, : self.seq_len, :],
        )

    def rollback(self, to_pos):
        """Truncate cache to `to_pos` (for speculative decoding rejection)."""
        self.seq_len = to_pos

    def reset(self):
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.seq_len = 0

# Differential Attention — two maps subtracted to cancel noise
class DifferentialAttention(nn.Module):
    """Differential Attention (Microsoft DIFF Transformer, 2024).

    Instead of a single attention map, computes two maps A₁ and A₂ from
    split Q/K sub-heads, then outputs (A₁ − λ·A₂)·V.  The subtraction
    cancels common-mode attention noise — analogous to noise-cancelling
    headphones.  λ is a learnable, reparameterized scalar that controls
    how aggressively noise is suppressed.

    Each layer is either *local* (sliding window + sinks always visible)
    or *global* (full causal).  The last layer in every stage is global;
    all others are local.
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.sub_head_dim = self.head_dim // 2  # Q/K each split in half

        # Standard linear projections (same total size as regular MHA)
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Lambda reparameterization — 4 learned vectors whose dot products
        # are exponentiated and combined to produce a stable scalar λ.
        self.lambda_q1 = nn.Parameter(torch.randn(self.sub_head_dim) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.sub_head_dim) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.sub_head_dim) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.sub_head_dim) * 0.1)

        # Layer-dependent init: deeper layers cancel more aggressively
        self.lambda_init = config.lambda_init - 0.6 * math.exp(-0.3 * layer_idx)

        # Per-head output norm (DIFF Transformer V1) — normalizes the
        # differential output which can have negative values and varying
        # magnitudes.  Different from QK-Norm (which we dropped).
        self.head_norm = RMSNorm(self.head_dim)

        # Local vs. global: last layer in each stage is global
        layers_per_stage = config.n_layer // config.n_stage
        pos_in_stage = layer_idx % layers_per_stage
        self.is_global = (pos_in_stage == layers_per_stage - 1)

        self.window_size = config.window_size
        self.n_sinks = config.n_sinks
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _build_mask(self, T_q, T_kv, device):
        """Build attention mask: causal + optional sliding window with sink exemption.

        Returns:
            Bool tensor to pass to masked_fill (True = masked out), or None
            if no masking is needed (single-token global attention).
        """
        if T_q == 1:
            # Single-token cached generation
            if not self.is_global:
                # Local: mask beyond window, but keep sinks visible
                k_pos = torch.arange(T_kv, device=device)
                is_sink = k_pos < self.n_sinks
                distance = (T_kv - 1) - k_pos
                return (distance >= self.window_size) & ~is_sink
            return None  # Global, single token: nothing to mask

        if self.is_global:
            # Standard upper-triangular causal mask
            return torch.triu(
                torch.ones(T_q, T_kv, dtype=torch.bool, device=device),
                diagonal=T_kv - T_q + 1,
            )
        else:
            # Local: causal + window limit, sinks always visible
            q_pos = torch.arange(T_q, device=device).unsqueeze(1) + (T_kv - T_q)
            k_pos = torch.arange(T_kv, device=device).unsqueeze(0)
            is_sink = k_pos < self.n_sinks
            distance = q_pos - k_pos
            future = distance < 0
            beyond_window = distance >= self.window_size
            return future | (beyond_window & ~is_sink)

    def update_cache_only(self, x, freqs_cis, kv_cache):
        """Minimal forward pass to update KV cache during state propagation."""
        B, T, _ = x.shape
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k1, k2 = k.chunk(2, dim=-1)
        dummy_q = torch.empty_like(k1)
        _, k1 = apply_rotary_emb(dummy_q, k1, freqs_cis)
        _, k2 = apply_rotary_emb(dummy_q, k2, freqs_cis)

        k_full = torch.cat([k1, k2], dim=-1)
        if kv_cache is not None:
            kv_cache.update(k_full, v)

    def forward(self, x, freqs_cis, kv_cache=None):
        B, T, _ = x.shape

        # Project Q, K, V → (B, n_head, T, head_dim)
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Split Q, K into sub-heads for the two attention maps
        q1, q2 = q.chunk(2, dim=-1)  # each (B, n_head, T, sub_head_dim)
        k1, k2 = k.chunk(2, dim=-1)

        # Apply RoPE to both sub-head pairs
        q1, k1 = apply_rotary_emb(q1, k1, freqs_cis)
        q2, k2 = apply_rotary_emb(q2, k2, freqs_cis)

        # Reconstruct full K for caching, then update cache
        k_full = torch.cat([k1, k2], dim=-1)  # (B, n_head, T, head_dim)
        if kv_cache is not None:
            k_full, v = kv_cache.update(k_full, v)
            # Re-split the full cached K into sub-heads
            k1, k2 = k_full.chunk(2, dim=-1)

        # Compute two attention maps
        T_q, T_kv = q1.shape[2], k1.shape[2]
        att1 = (q1 @ k1.transpose(-2, -1)) / (self.sub_head_dim ** 0.5)  # (B, n_head, T_q, T_kv)
        att2 = (q2 @ k2.transpose(-2, -1)) / (self.sub_head_dim ** 0.5)

        # Apply mask (causal + local window + sink exemption)
        mask = self._build_mask(T_q, T_kv, x.device)
        if mask is not None:
            att1 = att1.masked_fill(mask, float('-inf'))
            att2 = att2.masked_fill(mask, float('-inf'))

        # Softmax both maps
        att1 = self.attn_dropout(F.softmax(att1, dim=-1))
        att2 = self.attn_dropout(F.softmax(att2, dim=-1))

        # Compute lambda (reparameterized for stability)
        lambda_val = (
            torch.exp(self.lambda_q1 @ self.lambda_k1)
            - torch.exp(self.lambda_q2 @ self.lambda_k2)
            + self.lambda_init
        )

        # Differential attention: subtract and apply to values
        # (A_1 − lambda*A_2) can be negative to suppress distractors.
        out = (att1 - lambda_val * att2) @ v  # (B, n_head, T_q, head_dim)

        # Per-head norm + scaling (DIFF Transformer V1)
        out = self.head_norm(out) * (1 - self.lambda_init)

        # Reshape and output projection
        out = out.transpose(1, 2).contiguous().view(B, T_q, -1)
        return self.resid_dropout(self.wo(out))

# SwiGLU Feed-Forward — gated FFN with SiLU activation, 3 projections
class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0, bias=False):
        super().__init__()
        hidden_dim = hidden_dim or int(2 / 3 * 4 * dim)
        hidden_dim = 8 * ((hidden_dim + 7) // 8)  # align to 8 for hardware
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)   # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)    # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)    # up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# Transformer Block — pre-norm, differential attention + SwiGLU
class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = DifferentialAttention(config, layer_idx)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config.n_embd, dropout=config.dropout, bias=config.bias)

    def update_cache_only(self, x, freqs_cis, kv_cache):
        self.attn.update_cache_only(self.attn_norm(x), freqs_cis, kv_cache)

    def forward(self, x, freqs_cis, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x

# Exit Head — halting gate + LM logits + MTP at stage boundaries
class ExitHead(nn.Module):
    """Lightweight exit attached after each stage (except the last).

    Produces:
    - LM logits via a small adapter feeding into the shared output matrix
      (so exit heads add almost no embedding parameters)
    - A halting confidence score per token (sigmoid of a small MLP)
    - MTP logits predicting t+2 (training only — discarded at inference)
    """

    def __init__(self, config, output_weight):
        super().__init__()
        self.norm = RMSNorm(config.n_embd)
        # Small adapter to align exit representation to output space
        self.adapter = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # Shared output projection (reference, not a copy — no new params)
        self.output_weight = output_weight

        # Halting gate: n_embd → exit_hidden → 1 → sigmoid
        self.halt_mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.exit_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(config.exit_hidden, 1, bias=False),
        )

        # MTP: predict t+2 token (training only, discarded at inference)
        self.mtp_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, T, n_embd) — hidden states for real tokens (sinks stripped)
        Returns:
            logits:     (B, T, vocab_size) — next-token prediction
            confidence: (B, T) — halting confidence in [0, 1]
            mtp_logits: (B, T, vocab_size) — t+2 prediction (training only)
        """
        h = self.norm(x)
        logits = F.linear(self.adapter(h), self.output_weight)  # shared weights
        confidence = torch.sigmoid(self.halt_mlp(h)).squeeze(-1)
        mtp_logits = self.mtp_head(h)
        return logits, confidence, mtp_logits

# Transformer — stage-based orchestrator with multi-exit losses
class Transformer(nn.Module):
    """Waypoint adaptive-depth transformer with differential attention.

    Architecture:
    - n_layer layers grouped into n_stage stages
    - After each stage (except the last), an ExitHead provides:
      - Next-token logits (for early exit at inference)
      - Halting confidence (calibrated post-training)
      - MTP auxiliary loss (stronger gradient signal to early layers)
    - During training: all stages always run (dense), losses at every exit
    - During inference: optionally exit early when confident
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_layer % config.n_stage == 0, \
            f"n_layer ({config.n_layer}) must be divisible by n_stage ({config.n_stage})"
        assert len(config.exit_weights) == config.n_stage, \
            f"exit_weights length ({len(config.exit_weights)}) must equal n_stage ({config.n_stage})"

        self.layers_per_stage = config.n_layer // config.n_stage

        # --- Embeddings ---
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Learned sink tokens — prepended to every sequence, always visible
        # in local attention.  They absorb "dump" attention mass that would
        # otherwise distort the first few real tokens.
        self.sink_tokens = nn.Parameter(
            torch.randn(1, config.n_sinks, config.n_embd) * 0.02
        )

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layer)
        ])

        # --- Final output (last stage — always emits, no halting gate) ---
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding and output share the same matrix
        self.tok_emb.weight = self.output.weight

        # --- Exit heads (one per stage, except the last) ---
        self.exit_heads = nn.ModuleList([
            ExitHead(config, self.output.weight)
            for _ in range(config.n_stage - 1)
        ])

        # --- RoPE frequencies ---
        # dim = sub_head_dim = head_dim // 2  (for differential attention)
        # length = block_size + n_sinks  (to cover sink positions)
        head_dim = config.n_embd // config.n_head
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim // 2, config.block_size + config.n_sinks),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            # Scale down residual projections by 1/sqrt(2 * n_layer)
            # to prevent activation growth in deep networks
            if hasattr(module, '_is_residual'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        """Forward pass with multi-exit loss computation.

        During training: all stages run (dense training).  Losses are
        computed at every exit and weighted according to exit_weights.
        During inference (targets=None): all stages still run by default;
        use generate() with halt_threshold for early exit.

        Args:
            idx:       (B, T) token indices
            targets:   (B, T) next-token targets, or None for inference
            kv_caches: list of KVCache (one per layer), or None

        Returns:
            final_logits: (B, T, vocab_size) from the last stage
            total_loss:   weighted sum of all exit losses (scalar), or None
            exit_info:    list of dicts with keys:
                          logits, confidence, loss, stage
        """
        B, T = idx.shape

        # --- Embed ---
        x = self.tok_emb(idx)                                 # (B, T, n_embd)
        is_cached_step = kv_caches is not None and kv_caches[0].seq_len > 0

        # Prepend sink tokens only if we aren't continuing a cached sequence
        if not is_cached_step:
            sinks = self.sink_tokens.expand(B, -1, -1)        # (B, n_sinks, n_embd)
            x = torch.cat([sinks, x], dim=1)                  # (B, n_sinks+T, n_embd)
        T_full = x.shape[1]

        # --- RoPE slice ---
        if is_cached_step:
            start = kv_caches[0].seq_len
            freqs = self.freqs_cis[start : start + T_full]
        else:
            freqs = self.freqs_cis[:T_full]

        # --- MTP targets: t+2 prediction (shift targets by 1 more) ---
        mtp_targets = None
        if targets is not None and T > 1:
            mtp_targets = targets[:, 1:]                       # (B, T-1)

        exit_info = []
        total_loss = torch.tensor(0.0, device=idx.device) if targets is not None else None

        # --- Stage-by-stage forward ---
        for stage_idx in range(self.config.n_stage):
            start_layer = stage_idx * self.layers_per_stage
            end_layer = start_layer + self.layers_per_stage

            # Run all layers in this stage
            for layer_idx in range(start_layer, end_layer):
                cache = kv_caches[layer_idx] if kv_caches else None
                x = self.blocks[layer_idx](x, freqs, cache)

            # --- Exit head (every stage except the last) ---
            if stage_idx < self.config.n_stage - 1:
                # Strip sink positions — exit heads only see real tokens
                h_real = x if is_cached_step else x[:, self.config.n_sinks :, :]
                exit_logits, confidence, mtp_logits = self.exit_heads[stage_idx](h_real)

                exit_loss = None
                if targets is not None:
                    # t+1 cross-entropy loss
                    ce = F.cross_entropy(
                        exit_logits.view(-1, exit_logits.size(-1)),
                        targets.view(-1),
                    )
                    # t+2 MTP auxiliary loss
                    mtp_loss = torch.tensor(0.0, device=idx.device)
                    if mtp_targets is not None and mtp_targets.numel() > 0:
                        mtp_loss = F.cross_entropy(
                            mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1),
                        )
                    exit_loss = ce + self.config.mtp_weight * mtp_loss
                    total_loss = total_loss + self.config.exit_weights[stage_idx] * exit_loss

                exit_info.append({
                    'logits': exit_logits,
                    'confidence': confidence,
                    'loss': exit_loss,
                    'stage': stage_idx,
                })

        # --- Final output (last stage — always emits) ---
        h_real = x if is_cached_step else x[:, self.config.n_sinks :, :]
        final_logits = self.output(self.norm(h_real))          # (B, T, vocab_size)

        final_loss = None
        if targets is not None:
            final_loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)),
                targets.view(-1),
            )
            total_loss = total_loss + self.config.exit_weights[-1] * final_loss

        exit_info.append({
            'logits': final_logits,
            'confidence': None,   # final stage always emits — no halting gate
            'loss': final_loss,
            'stage': self.config.n_stage - 1,
        })

        return final_logits, total_loss, exit_info

    def create_kv_caches(self, batch_size=1, device=None):
        """Create one KV cache per layer for autoregressive generation."""
        device = device or next(self.parameters()).device
        head_dim = self.config.n_embd // self.config.n_head
        max_len = self.config.block_size + self.config.n_sinks
        return [
            KVCache(batch_size, max_len, self.config.n_head, head_dim, device)
            for _ in range(self.config.n_layer)
        ]

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,
                 use_cache=True, halt_threshold=None):
        """Autoregressive generation using Self-Speculative Sampling.
        
        Args:
            idx:            (1, prompt_len) token indices
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (e.g. 0.8)
            use_cache:      must be True for speculative decoding
            halt_threshold: if provided, uses early exit during the verification step
        Returns:
            idx: (1, prompt_len + generated) full sequence
        """
        device = idx.device
        kv_caches = self.create_kv_caches(batch_size=idx.shape[0], device=device)
        lps = self.layers_per_stage
        
        # Warm cache with prompt
        logits, _, _ = self(idx, kv_caches=kv_caches)
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        first_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, first_token], dim=1)
        generated = 1
        
        while generated < max_new_tokens:
            cache_pos = kv_caches[0].seq_len
            last_token = idx[:, -1:]
            
            # === DRAFT: Stage 0 with cache, then rollback ===
            x = self.tok_emb(last_token)
            freqs = self.freqs_cis[cache_pos : cache_pos + 1]
            for li in range(lps):
                x = self.blocks[li](x, freqs, kv_caches[li])
            draft_logits, _, _ = self.exit_heads[0](x)
            
            # Draft Sampling
            draft_probs = F.softmax(draft_logits[:, -1, :] / temperature, dim=-1)
            draft_token = torch.multinomial(draft_probs, num_samples=1)
            
            # Rollback Stage 0 caches (verify will redo this)
            for li in range(lps):
                kv_caches[li].rollback(cache_pos)
            
            # === VERIFY ===
            verify_ids = torch.cat([last_token, draft_token], dim=1)
            
            if halt_threshold is not None:
                # Early Exit Verification
                x = self.tok_emb(verify_ids)
                start = kv_caches[0].seq_len
                freqs = self.freqs_cis[start : start + x.shape[1]]
                
                full_logits = None
                for stage_idx in range(self.config.n_stage):
                    s = stage_idx * self.layers_per_stage
                    e = s + self.layers_per_stage
                    for li in range(s, e):
                        x = self.blocks[li](x, freqs, kv_caches[li])
                    
                    if stage_idx < self.config.n_stage - 1:
                        logits, confidence, _ = self.exit_heads[stage_idx](x)
                        if confidence[:, -1].item() > halt_threshold:
                            for rem in range(e, self.config.n_layer):
                                self.blocks[rem].update_cache_only(x, freqs, kv_caches[rem])
                            full_logits = logits
                            break
                if full_logits is None:
                    full_logits = self.output(self.norm(x))
            else:
                # Full depth verification
                full_logits, _, _ = self(verify_ids, kv_caches=kv_caches)
            
            # Verify probabilities
            verify_probs = F.softmax(full_logits[:, 0, :] / temperature, dim=-1)
            
            # Calculate acceptance probability
            p = verify_probs[0, draft_token[0, 0]]
            q = draft_probs[0, draft_token[0, 0]]
            
            r = torch.rand(1, device=device).item()
            if r < (p / q).item():
                # ACCEPT
                bonus_probs = F.softmax(full_logits[:, 1, :] / temperature, dim=-1)
                bonus = torch.multinomial(bonus_probs, num_samples=1)
                idx = torch.cat([idx, draft_token, bonus], dim=1)
                generated += 2
            else:
                # REJECT: Resample from max(0, p(x) - q(x))
                new_probs = torch.clamp(verify_probs - draft_probs, min=0.0)
                new_probs = new_probs / new_probs.sum(dim=-1, keepdim=True)
                true_next = torch.multinomial(new_probs, num_samples=1)
                
                for cache in kv_caches:
                    cache.rollback(cache_pos + 1)
                idx = torch.cat([idx, true_next], dim=1)
                generated += 1
                
        return idx
