"""
Speculative Decoding Benchmark
===============================
Benchmarks speculative decoding against standard autoregressive generation
using pretrained HuggingFace GPT-2 models. No training required.

Target: gpt2-large  (774M params)
Draft:  distilgpt2  (82M params)

Methodology:
  - CUDA warmup runs before timing
  - torch.cuda.synchronize() at timing boundaries
  - Multiple trials with mean +/- std reporting
  - Gamma sweep across multiple speculation depths

Usage:
  pip install transformers accelerate
  python benchmark.py
"""

import torch
import torch.nn.functional as F
import time
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer

# NOTE: This triggers gpt.py's module-level data loading as a side effect.
# This will be resolved when model.py replaces gpt.py in the pipeline refactor.
from speculative import speculative_generate

# ============================================================================
# Configuration
# ============================================================================
TARGET_MODEL = "gpt2-large"       # 774M params — the "expensive" model
DRAFT_MODEL  = "distilgpt2"       # 82M params  — the "cheap" guesser
GAMMA_VALUES = [2, 4, 6, 8]       # speculation depths to sweep
MAX_NEW_TOKENS = 512
NUM_WARMUP = 3                    # untimed warmup runs (CUDA initialization)
NUM_TRIALS = 10                   # timed measurement runs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = "The future of artificial intelligence is"


def sync():
    """Block until all GPU kernels have finished (no-op on CPU)."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ============================================================================
# Model Loading
# ============================================================================
def load_models():
    """Load pretrained target and draft models. No training needed."""
    print(f"  Loading target: {TARGET_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    print(f"  Loading draft:  {DRAFT_MODEL} ...")
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_params = sum(p.numel() for p in target.parameters())
    draft_params  = sum(p.numel() for p in draft.parameters())
    print(f"  Target: {target_params / 1e6:.0f}M params")
    print(f"  Draft:  {draft_params / 1e6:.0f}M params")
    print(f"  Ratio:  {target_params / draft_params:.1f}x\n")

    return target, draft, tokenizer


# ============================================================================
# Standard Autoregressive Generation
# ============================================================================
@torch.no_grad()
def standard_generate(model, input_ids, max_new_tokens):
    """
    Baseline: generate one token at a time using only the target model.
    Each step runs a full forward pass (no KV cache) to keep the comparison
    fair with the speculative method.
    """
    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(idx, use_cache=False).logits[:, -1, :]   # (1, vocab)
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return idx


# ============================================================================
# Benchmark Harness
# ============================================================================
def run_standard(model, input_ids):
    """Time standard generation with warmup and multiple trials."""
    print(f"  Warmup ({NUM_WARMUP} runs)...", end=" ", flush=True)
    for _ in range(NUM_WARMUP):
        standard_generate(model, input_ids.clone(), MAX_NEW_TOKENS)
    print("done")

    tok_sec_list = []
    for i in range(NUM_TRIALS):
        sync()
        t0 = time.perf_counter()
        out = standard_generate(model, input_ids.clone(), MAX_NEW_TOKENS)
        sync()
        elapsed = time.perf_counter() - t0

        n = out.shape[1] - input_ids.shape[1]
        tps = n / elapsed
        tok_sec_list.append(tps)
        print(f"    [{i+1:>2}/{NUM_TRIALS}]  {elapsed:>6.2f}s  "
              f"{n:>4} tok  {tps:>7.1f} tok/s")

    return tok_sec_list


def run_speculative(target, draft, input_ids, gamma):
    """Time speculative generation with warmup and multiple trials."""
    # Wrap HuggingFace models to match speculative_generate's interface:
    # callable(idx) -> logits tensor of shape (B, T, vocab_size)
    target_fn = lambda idx: target(idx, use_cache=False).logits
    draft_fn = lambda idx: draft(idx, use_cache=False).logits

    print(f"  Warmup ({NUM_WARMUP} runs)...", end=" ", flush=True)
    for _ in range(NUM_WARMUP):
        speculative_generate(target_fn, draft_fn, input_ids.clone(),
                             MAX_NEW_TOKENS, gamma)
    print("done")

    tok_sec_list = []
    acc_list = []
    for i in range(NUM_TRIALS):
        sync()
        t0 = time.perf_counter()
        out, acc = speculative_generate(target_fn, draft_fn, input_ids.clone(),
                                        MAX_NEW_TOKENS, gamma)
        sync()
        elapsed = time.perf_counter() - t0

        n = out.shape[1] - input_ids.shape[1]
        tps = n / elapsed
        tok_sec_list.append(tps)
        acc_list.append(acc)
        print(f"    [{i+1:>2}/{NUM_TRIALS}]  {elapsed:>6.2f}s  "
              f"{n:>4} tok  {tps:>7.1f} tok/s  accept {acc:.1%}")

    return tok_sec_list, acc_list


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    banner = "SPECULATIVE DECODING BENCHMARK"
    print("\n" + "=" * 70)
    print(f"  {banner}")
    print("=" * 70)
    print(f"  Target model : {TARGET_MODEL}")
    print(f"  Draft model  : {DRAFT_MODEL}")
    print(f"  New tokens   : {MAX_NEW_TOKENS}")
    print(f"  Gamma sweep  : {GAMMA_VALUES}")
    print(f"  Trials       : {NUM_TRIALS}  (+{NUM_WARMUP} warmup)")
    print(f"  Device       : {DEVICE}")
    print(f"  Precision    : {DTYPE}")
    print("=" * 70 + "\n")

    # --- Load models ---
    target_model, draft_model, tokenizer = load_models()
    input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    print(f"  Prompt: \"{PROMPT}\"  ({input_ids.shape[1]} tokens)\n")

    # --- Baseline: standard autoregressive ---
    print("=" * 70)
    print("  BASELINE — Standard Autoregressive")
    print("=" * 70)
    std_tps = run_standard(target_model, input_ids)
    std_mean = statistics.mean(std_tps)
    std_std  = statistics.stdev(std_tps) if len(std_tps) > 1 else 0.0
    print(f"\n  ► Standard: {std_mean:.1f} ± {std_std:.1f} tok/s\n")

    # --- Gamma sweep ---
    results = {}
    for gamma in GAMMA_VALUES:
        print("=" * 70)
        print(f"  SPECULATIVE DECODING — γ = {gamma}")
        print("=" * 70)
        spec_tps, acc_rates = run_speculative(
            target_model, draft_model, input_ids, gamma
        )
        spec_mean = statistics.mean(spec_tps)
        spec_std  = statistics.stdev(spec_tps) if len(spec_tps) > 1 else 0.0
        acc_mean  = statistics.mean(acc_rates)
        acc_std   = statistics.stdev(acc_rates) if len(acc_rates) > 1 else 0.0

        results[gamma] = {
            "tps_mean": spec_mean,  "tps_std": spec_std,
            "acc_mean": acc_mean,   "acc_std": acc_std,
            "speedup": spec_mean / std_mean,
        }
        r = results[gamma]
        print(f"\n  ► γ={gamma}: {r['tps_mean']:.1f} ± {r['tps_std']:.1f} tok/s  "
              f"| accept {r['acc_mean']:.1%}  | {r['speedup']:.2f}x\n")

    # ================================================================
    #  Summary Table
    # ================================================================
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    hdr = f"  {'Method':<24}{'Tokens/sec':>14}  {'Acceptance':>14}  {'Speedup':>8}"
    print(hdr)
    print("  " + "-" * 64)
    print(f"  {'Standard':<24}"
          f"{std_mean:>7.1f} ± {std_std:<4.1f}  "
          f"{'—':>14}  {'1.00x':>8}")

    best_gamma, best_speedup = None, 0.0
    for g in GAMMA_VALUES:
        r = results[g]
        tps_str = f"{r['tps_mean']:.1f} ± {r['tps_std']:.1f}"
        acc_str = f"{r['acc_mean']:.1%} ± {r['acc_std']:.1%}"
        spd_str = f"{r['speedup']:.2f}x"
        print(f"  {'Speculative (γ=' + str(g) + ')':<24}"
              f"{tps_str:>14}  {acc_str:>14}  {spd_str:>8}")
        if r["speedup"] > best_speedup:
            best_speedup = r["speedup"]
            best_gamma = g

    # ================================================================
    #  Best Result
    # ================================================================
    b = results[best_gamma]
    print("  " + "-" * 64)
    print(f"\n  ★  Best configuration: γ = {best_gamma}")
    print(f"     {b['speedup']:.2f}x speedup  |  "
          f"{b['tps_mean']:.0f} tok/s (vs {std_mean:.0f} baseline)  |  "
          f"{b['acc_mean']:.0%} acceptance")
    print()
