# Custom Adaptive-Inference Transformer LLM

This repository contains an experimental Large Language Model architecture built from scratch in PyTorch, designed to investigate inference efficiency through a combination of **Early Exiting** and **Self-Speculative Decoding**.

## 🚀 Key Innovations

### 1. Early Exiting (Adaptive Depth)
Standard LLMs run every token through every layer. This project attaches "Exit Heads" at intermediate stages of the network. If an early stage is highly confident in its prediction (e.g., predicting a common word like "the"), the model halts computation and outputs the token immediately, skipping the remaining layers.

### 2. Self-Speculative Decoding
Traditional speculative decoding requires a separate, smaller "draft" model. This project implements **Self-Speculation** using Stage 0 (the first few layers of the main model) to act as the draft model. 
- **Drafting**: Stage 0 rapidly predicts the next token.
- **Verification**: The full model verifies the draft while simultaneously predicting the *next* token.
- **Result**: If accepted, the model outputs 2 tokens in a single step, overcoming memory bandwidth bottlenecks.

### 3. Combined: Speculative + Early Exit Verification
A core feature of this architecture: Stage 0 drafts a token, and the verifier runs stage-by-stage. If the verifier becomes confident early, it exits, verifying the draft *without* running the full depth of the network. This yields massive speedups (e.g., 1.85x over baseline).

### 4. Advanced Architecture Details
- **Differential Attention**: Enhances multi-head attention by subtracting attention scores, reducing noise and focusing on critical context.
- **Multi-Token Prediction (MTP)**: Auxiliary loss during training forces exit heads to predict the next token *and* the token after that, encouraging deeper feature extraction earlier in the network.
- **KV Cache Rollback**: Custom KV cache implementation that supports instant rollbacks for rejected speculative drafts.
- **Custom BPE Tokenizer**: Built from scratch using the `regex` and `collections` libraries.

## 📊 Benchmarks

Run `benchmark_architecture.py` to see the real-world impact. Example results on a 100M parameter model:

| Method | Speedup | How it works |
|--------|---------|--------------|
| **1. Baseline** | 1.00x | Full depth, standard autoregressive generation |
| **2. Early Exit** | 1.48x | Stops at early stages when confident |
| **3. Self-Speculative** | 1.22x | Stage 0 drafts, Full Model verifies |
| **4. Combined** | **1.85x** | Stage 0 drafts, Early Exit verifies |

*(Note: Speculative speedups scale significantly with larger parameter counts and longer training)*

## 🛠️ Usage

### 1. Train the Tokenizer
```bash
python tokenizer.py
```
*Trains on the text dataset and generates `data/tokenizer.json` and `data/tokenizer.vocab`.*

### 2. Train the Model
```bash
python train.py
```
*Trains the model using cross-entropy and MTP loss. Checkpoints are saved to `data/checkpoint.pt`.*

### 3. Interactive Generation
```bash
python generate.py
```
*Chat with the model. You can manually adjust the `halt_threshold` to see how Early Exiting affects speed and output quality.*

### 4. Run Benchmarks
```bash
python benchmark_architecture.py
```

## 🧠 File Structure

- `model.py` — The core Transformer, Differential Attention, Exit Heads, and KV Cache.
- `config.py` — Hyperparameters and model configuration.
- `train.py` — Training loop with loss weighting across multiple exit stages.
- `generate.py` — Interactive CLI for text generation.
- `benchmark_architecture.py` — Inference speed benchmarking.
- `speculative.py` — Implementation of the speculative decoding loops.
- `tokenizer.py` — Custom Byte-Pair Encoding (BPE) implementation.
