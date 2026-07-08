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
def baseline_generate(model, prompt, gen_len, halt_threshold=None):
    """Standard token-by-token generation for baseline testing."""
    kv_caches = model.create_kv_caches(batch_size=1, device=prompt.device)
    idx = prompt.clone()
    
    for _ in range(gen_len):
        if kv_caches[0].seq_len > 0:
            idx_cond = idx[:, -1:]
        else:
            idx_cond = idx
            
        if halt_threshold is None:
            logits, _, _ = model(idx_cond, kv_caches=kv_caches)
        else:
            # Reimplement adaptive depth for benchmark baseline
            x = model.tok_emb(idx_cond)
            is_cached = kv_caches[0].seq_len > 0
            if not is_cached:
                sinks = model.sink_tokens.expand(1, -1, -1)
                x = torch.cat([sinks, x], dim=1)
            
            start = kv_caches[0].seq_len if is_cached else 0
            freqs = model.freqs_cis[start : start + x.shape[1]]
            
            logits = None
            for stage_idx in range(model.config.n_stage):
                s = stage_idx * model.layers_per_stage
                e = s + model.layers_per_stage
                for li in range(s, e):
                    x = model.blocks[li](x, freqs, kv_caches[li])
                
                if stage_idx < model.config.n_stage - 1:
                    h_real = x if is_cached else x[:, model.config.n_sinks:, :]
                    out_logits, conf, _ = model.exit_heads[stage_idx](h_real)
                    if conf[:, -1].item() > halt_threshold:
                        for rem in range(e, model.config.n_layer):
                            model.blocks[rem].update_cache_only(x, freqs, kv_caches[rem])
                        logits = out_logits
                        break
            if logits is None:
                h_real = x if is_cached else x[:, model.config.n_sinks:, :]
                logits = model.output(model.norm(h_real))
                
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, idx_next], dim=1)
        
    return idx

@torch.no_grad()
def benchmark(model, config, device, gen_len=100):
    prompt = torch.randint(0, config.vocab_size, (1, 32), device=device)
    
    # Warmup
    print("Warming up...", end=" ", flush=True)
    baseline_generate(model, prompt.clone(), 5, halt_threshold=None)
    model.generate(prompt.clone(), 5, halt_threshold=0.5)
    print("done.\n")
    
    # ---- 1. Baseline: Full Depth ----
    sync()
    t0 = time.perf_counter()
    baseline_generate(model, prompt.clone(), gen_len, halt_threshold=None)
    sync()
    t_baseline = time.perf_counter() - t0
    
    # ---- 2. Early Exit Only ----
    sync()
    t0 = time.perf_counter()
    baseline_generate(model, prompt.clone(), gen_len, halt_threshold=0.5)
    sync()
    t_early = time.perf_counter() - t0
    
    # ---- 3. Self-Speculative Only (full depth verify) ----
    sync()
    t0 = time.perf_counter()
    out = model.generate(prompt.clone(), gen_len, halt_threshold=None)
    sync()
    t_spec = time.perf_counter() - t0
    gen_spec = out.shape[1] - prompt.shape[1]
    
    # ---- 4. Combined: Self-Spec + Early Exit verify ----
    sync()
    t0 = time.perf_counter()
    out = model.generate(prompt.clone(), gen_len, halt_threshold=0.5)
    sync()
    t_comb = time.perf_counter() - t0
    gen_comb = out.shape[1] - prompt.shape[1]
    
    # ---- Results ----
    tps_baseline = gen_len / t_baseline
    tps_early    = gen_len / t_early
    tps_spec     = gen_spec / t_spec
    tps_comb     = gen_comb / t_comb
    
    print(f"{'Method':<40} {'tok/s':>8} {'Speedup':>8}")
    print("-" * 60)
    print(f"{'1. Baseline (Full Depth)':<40} {tps_baseline:>8.1f} {1.0:>8.2f}x")
    print(f"{'2. Early Exit Only':<40} {tps_early:>8.1f} {tps_early/tps_baseline:>8.2f}x")
    print(f"{'3. Self-Speculative Only':<40} {tps_spec:>8.1f} {tps_spec/tps_baseline:>8.2f}x")
    print(f"{'4. Self-Spec + Early Exit':<40} {tps_comb:>8.1f} {tps_comb/tps_baseline:>8.2f}x")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on: {device.upper()}")
    
    try:
        model, cfg = load_model(device)
    except FileNotFoundError as e:
        print(e)
        exit(1)
    
    benchmark(model, cfg, device, gen_len=100)
