"""
Architecture Benchmark
======================
Loads the trained checkpoint and honestly measures inference speed across
four configurations to show the real-world impact of each optimization:

1. Baseline          — Full depth, KV cached generation
2. Early Exit        — Adaptive depth (halt_threshold=0.5), KV cached
3. Self-Speculative  — Stage 0 drafts + full verify, KV cached with rollback
4. Combined          — Stage 0 drafts + early-exit verify, KV cached with rollback
"""

import os
import time
import torch
from config import ModelConfig
from model import Transformer

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def load_model(device, ckpt_path="data/checkpoint.pt"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Train the model first with train.py."
        )
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', ModelConfig())
    model = Transformer(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg


def verify_early_exit(model, verify_ids, kv_caches, halt_threshold=0.5):
    """Verify draft tokens stage-by-stage with early exit.
    
    Returns (B, T, vocab_size) logits — same shape as model.forward(),
    but may exit early if an exit head is confident enough.
    """
    x = model.tok_emb(verify_ids)
    start = kv_caches[0].seq_len
    freqs = model.freqs_cis[start : start + x.shape[1]]
    
    for stage_idx in range(model.config.n_stage):
        s = stage_idx * model.layers_per_stage
        e = s + model.layers_per_stage
        for li in range(s, e):
            x = model.blocks[li](x, freqs, kv_caches[li])
        
        # Check exit at every stage except the last
        if stage_idx < model.config.n_stage - 1:
            logits, confidence, _ = model.exit_heads[stage_idx](x)
            if confidence[:, -1].item() > halt_threshold:
                # Propagate state to remaining layer caches
                for rem in range(e, model.config.n_layer):
                    model.blocks[rem].update_cache_only(x, freqs, kv_caches[rem])
                return logits  # (B, T, vocab_size)
    
    # Final stage — always emits
    return model.output(model.norm(x))


@torch.no_grad()
def speculative_generate(model, prompt, gen_len, device, use_early_exit=False):
    """Self-speculative generation with KV cache and rollback.
    
    Draft:  Run 1 token through Stage 0 layers only (with cache, then rollback).
    Verify: Run [last_token, draft_token] through full model (with cache).
    Accept: Keep both cached entries + get a bonus token (2 tokens per step).
    Reject: Rollback cache to before draft, keep the corrected token (1 token per step).
    """
    kv_caches = model.create_kv_caches(batch_size=1, device=device)
    lps = model.layers_per_stage
    
    # Warm cache with prompt
    logits, _, _ = model(prompt, kv_caches=kv_caches)
    first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    idx = torch.cat([prompt, first_token], dim=1)
    generated = 1
    accepted = 0
    
    while generated < gen_len:
        cache_pos = kv_caches[0].seq_len
        last_token = idx[:, -1:]
        
        # === DRAFT: Stage 0 with cache, then rollback ===
        x = model.tok_emb(last_token)
        freqs = model.freqs_cis[cache_pos : cache_pos + 1]
        for li in range(lps):
            x = model.blocks[li](x, freqs, kv_caches[li])
        draft_logits, _, _ = model.exit_heads[0](x)
        draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
        
        # Rollback Stage 0 caches (verify will redo this)
        for li in range(lps):
            kv_caches[li].rollback(cache_pos)
        
        # === VERIFY ===
        verify_ids = torch.cat([last_token, draft_token], dim=1)
        if use_early_exit:
            full_logits = verify_early_exit(model, verify_ids, kv_caches)
        else:
            full_logits, _, _ = model(verify_ids, kv_caches=kv_caches)
        
        true_next = torch.argmax(full_logits[:, 0, :], dim=-1, keepdim=True)
        
        if true_next.item() == draft_token.item():
            accepted += 1
            bonus = torch.argmax(full_logits[:, 1, :], dim=-1, keepdim=True)
            idx = torch.cat([idx, draft_token, bonus], dim=1)
            generated += 2
        else:
            for cache in kv_caches:
                cache.rollback(cache_pos + 1)
            idx = torch.cat([idx, true_next], dim=1)
            generated += 1
    
    return idx, accepted, generated


@torch.no_grad()
def benchmark(model, config, device, gen_len=100):
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=device)
    
    # Warmup
    print("Warming up...", end=" ", flush=True)
    model.generate(prompt.clone(), 5, use_cache=True, halt_threshold=None)
    model.generate(prompt.clone(), 5, use_cache=True, halt_threshold=0.5)
    print("done.\n")
    
    # ---- 1. Baseline: Full Depth ----
    sync()
    t0 = time.perf_counter()
    model.generate(prompt.clone(), gen_len, use_cache=True, halt_threshold=None)
    sync()
    t_baseline = time.perf_counter() - t0
    
    # ---- 2. Early Exit Only ----
    sync()
    t0 = time.perf_counter()
    model.generate(prompt.clone(), gen_len, use_cache=True, halt_threshold=0.5)
    sync()
    t_early = time.perf_counter() - t0
    
    # ---- 3. Self-Speculative Only (full depth verify) ----
    sync()
    t0 = time.perf_counter()
    _, acc_spec, gen_spec = speculative_generate(
        model, prompt.clone(), gen_len, device, use_early_exit=False
    )
    sync()
    t_spec = time.perf_counter() - t0
    
    # ---- 4. Combined: Self-Speculative + Early Exit verify ----
    sync()
    t0 = time.perf_counter()
    _, acc_comb, gen_comb = speculative_generate(
        model, prompt.clone(), gen_len, device, use_early_exit=True
    )
    sync()
    t_comb = time.perf_counter() - t0
    
    # ---- Results ----
    tps_baseline = gen_len / t_baseline
    tps_early    = gen_len / t_early
    tps_spec     = gen_spec / t_spec
    tps_comb     = gen_comb / t_comb
    
    print(f"{'Method':<40} {'tok/s':>8} {'Speedup':>8} {'Accepted':>10}")
    print("-" * 70)
    print(f"{'1. Baseline (Full Depth)':<40} {tps_baseline:>8.1f} {1.0:>8.2f}x {'—':>10}")
    print(f"{'2. Early Exit Only':<40} {tps_early:>8.1f} {tps_early/tps_baseline:>8.2f}x {'—':>10}")
    print(f"{'3. Self-Speculative Only':<40} {tps_spec:>8.1f} {tps_spec/tps_baseline:>8.2f}x {f'{acc_spec}/{gen_spec}':>10}")
    print(f"{'4. Self-Spec + Early Exit':<40} {tps_comb:>8.1f} {tps_comb/tps_baseline:>8.2f}x {f'{acc_comb}/{gen_comb}':>10}")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on: {device.upper()}")
    
    try:
        model, cfg = load_model(device)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    benchmark(model, cfg, device, gen_len=100)
