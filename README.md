# Custom Adaptive-Inference Transformer LLM

This repository contains an experimental Large Language Model architecture built from scratch in PyTorch, designed to investigate inference efficiency through a combination of **Early Exiting** and **Self-Speculative Decoding**.

## 🚀 Model Architecture

The `generate.py` script natively implements the following unified pipeline:

1. **Self-Speculative Drafting**: Stage 0 (the first few layers of the model) acts as an internal draft model, rapidly predicting the next token.
2. **Rejection Sampling**: Instead of greedy decoding, the model uses Rejection Sampling to mathematically align the probability distributions of the draft and verifier models. This guarantees coherent text generation while preserving the speculative speedup.
3. **Early Exit Verification**: When verifying the draft, the model evaluates its confidence stage-by-stage. If a `halt_threshold` is provided and the verifier becomes confident early, it halts computation and verifies the draft *without* running the full depth of the network.
4. **Differential Attention**: Enhances multi-head attention by subtracting two attention maps, canceling background noise and focusing on critical context.
5. **Hybrid Sliding Window**: Within each stage, the first layers use local attention (256-token window + 4 learned sink tokens) for efficiency, while the last layer uses full causal attention for long-range coherence.
6. **Multi-Token Prediction (MTP)**: Auxiliary loss during training forces the exit heads to predict the next token *and* the token after that, forcing the early layers to become smarter faster.
7. **KV Cache Rollback**: Custom KV cache implementation that supports instant rollbacks for rejected speculative drafts.

## 📊 Benchmarks

Run `benchmark.py` to see the real-world impact. Example results on a 100M parameter model:

| Method | Speedup | How it works |
|--------|---------|--------------|
| **1. Baseline** | 1.00x | Full depth, standard autoregressive generation |
| **2. Early Exit** | 1.51x | Stops at early stages when confident |
| **3. Self-Speculative** | 1.37x | Stage 0 drafts, Full Model verifies |
| **4. Combined** | 2.29x | Stage 0 drafts, Early Exit verifies |

*(Note: Speculative speedups scale significantly with larger parameter counts and longer training)*

## 🛠️ Usage

### 1. Train the Tokenizer
```bash
python tokenizer.py
```
*Trains on the text dataset and generates `data/tokenizer.merges`.*

### 2. Train the Model
```bash
python train.py
```
*Trains the model using cross-entropy and MTP loss. Checkpoints are saved to `data/checkpoint.pt` (ignored by git).*

### 3. Interactive Generation
```bash
python generate.py
```
*Chat with the model. **This script natively uses the highly-optimized Self-Speculative Sampling engine by default.** You can also manually adjust the `halt_threshold` to enable Early Exiting on top of it.*

### 4. Run Benchmarks
```bash
python benchmark.py
```
*Tests all four generation configurations side-by-side to measure authentic token-per-second speedups.*

## 🧠 File Structure

- `model.py` — The core Transformer, Differential Attention, Exit Heads, KV Cache, and Speculative Sampling engine.
- `config.py` — Hyperparameters and model configuration.
- `train.py` — Training loop with loss weighting across multiple exit stages.
- `generate.py` — Interactive CLI for text generation (defaults to Speculative Sampling).
- `benchmark.py` — Inference speed benchmarking.
- `speculative.py` — An interactive, educational demo script breaking down how Self-Speculative decoding works step-by-step.
- `tokenizer.py` — Custom Byte-Pair Encoding (BPE) implementation.
- `pack_data.py` — Converts the raw text dataset into efficient `.pt` tensors for training.
- `scrape.py` — Utility script to scrape text data from the web.
- `test_model.py` & `test_overfit.py` — Smoke tests and mathematical sanity checks for the architecture.
- `data/` — Local data directory containing the tokenizer merge rules and demo text files. (Large raw text datasets and trained `.pt` checkpoints are hidden by `.gitignore`).
- `.gitignore` — Filters out large checkpoint files, system caches, and raw datasets from source control.
