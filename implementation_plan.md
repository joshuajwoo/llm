# End-to-End Custom LLM Pipeline

Rebuild the project so that **every component is yours** — from data scraping through tokenization, model architecture, training, and inference optimization. The goal is a project where you can walk an interviewer through each piece and explain *why* you made each design decision.

## Current State

| File | Status | Notes |
|------|--------|-------|
| [scrape.py](file:///c:/Users/joshu/OneDrive/projects/llm/scrape.py) | ✅ Custom | LaTeX-aware web scraper with multi-source math handling |
| [upload_to_hf.py](file:///c:/Users/joshu/OneDrive/projects/llm/upload_to_hf.py) | ✅ Custom | Dataset publishing |
| [speculative.py](file:///c:/Users/joshu/OneDrive/projects/llm/speculative.py) | ✅ Custom (just refactored) | Model-agnostic speculative decoding |
| [benchmark.py](file:///c:/Users/joshu/OneDrive/projects/llm/benchmark.py) | ✅ Fixed | Now imports from speculative.py |
| [gpt.py](file:///c:/Users/joshu/OneDrive/projects/llm/gpt.py) | ❌ Tutorial code | Karpathy's nanoGPT — needs replacement |

## Target File Structure

```
llm/
├── data/
│   └── input.txt              # Scraped corpus
├── checkpoints/               # Saved model weights
│   ├── target/
│   └── draft/
├── config.py                  # [NEW] All hyperparameters and model configs
├── tokenizer.py               # [NEW] Custom BPE tokenizer
├── model.py                   # [NEW] Custom transformer architecture
├── train.py                   # [NEW] Training pipeline with WandB
├── speculative.py             # [MODIFY] Update to use new model.py
├── benchmark.py               # [MODIFY] Benchmark custom models
├── scrape.py                  # [KEEP] Already custom
├── upload_to_hf.py            # [KEEP] Already custom
├── gpt.py                     # [DEPRECATE] Keep for reference, no longer imported
└── README.md                  # [REWRITE] Full project documentation
```

---

## Proposed Changes

### Phase 1 — Configuration

#### [NEW] [config.py](file:///c:/Users/joshu/OneDrive/projects/llm/config.py)

Centralize all hyperparameters using Python dataclasses. Every other file imports from here — no more magic numbers scattered across files.

```python
@dataclass
class ModelConfig:
    vocab_size: int = 4096        # Set after tokenizer training
    block_size: int = 512         # Context window length
    n_embd: int = 384             # Embedding dimension
    n_head: int = 6               # Number of attention heads
    n_kv_head: int = 2            # KV heads for GQA (n_head must be divisible by this)
    n_layer: int = 6              # Transformer blocks
    dropout: float = 0.1
    bias: bool = False            # Linear layer bias (modern LLMs skip this)

@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5          # For cosine schedule
    warmup_iters: int = 200
    eval_interval: int = 500
    eval_iters: int = 200
    grad_accum_steps: int = 1
    weight_decay: float = 0.1
    grad_clip: float = 1.0

@dataclass
class DraftConfig(ModelConfig):
    """Smaller config for the draft model used in speculative decoding."""
    n_embd: int = 64
    n_head: int = 2
    n_kv_head: int = 1
    n_layer: int = 1
    dropout: float = 0.0
```

> [!IMPORTANT]
> `vocab_size` gets set dynamically after the tokenizer is trained. Config should support loading/saving to JSON for reproducibility.

---

### Phase 2 — Custom BPE Tokenizer

#### [NEW] [tokenizer.py](file:///c:/Users/joshu/OneDrive/projects/llm/tokenizer.py)

Replace the character-level tokenizer (65 chars from Shakespeare) with a **byte-pair encoding tokenizer trained on your dataset**. This is the single biggest upgrade — it changes the model from a toy to something that handles real text.

**Implementation details:**

1. **Byte-level base vocabulary** (256 entries) — every possible byte is a token, so the tokenizer can handle any input without `UnicodeError`
2. **BPE merge loop** — iteratively find the most frequent adjacent token pair and merge them into a new token, until reaching `vocab_size`
3. **Special tokens**: `<|endoftext|>` (document separator, already in your scraped data), `<|pad|>`
4. **Encode/decode methods** — `encode(text) -> list[int]`, `decode(ids) -> str`
5. **Save/load** — serialize the merge table and vocab to disk so you don't retrain every time

```python
class BPETokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.merges = {}          # (pair) -> new_token_id
        self.vocab = {}           # token_id -> bytes
        self.special_tokens = {}  # string -> token_id
    
    def train(self, text: str):
        """Train BPE on corpus. Builds merge table greedily."""
        # 1. Convert text to list of byte tokens (UTF-8 encoded)
        # 2. While len(vocab) < vocab_size:
        #    a. Count all adjacent pairs
        #    b. Merge the most frequent pair into a new token
        #    c. Record the merge rule
    
    def encode(self, text: str) -> list[int]:
        """Encode string to token ids using learned merges."""
    
    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to string."""
    
    def save(self, path: str): ...
    def load(self, path: str): ...
```

> [!TIP]
> **Why this matters for interviews:** A BPE tokenizer is the #1 thing interviewers don't expect candidates to have built from scratch. It shows you understand the full pipeline, not just the transformer. The implementation is ~150 lines but touches compression theory, Unicode handling, and efficiency (use a heap for O(n log n) merges).

**Key design decisions:**
- **vocab_size = 4096** as a starting point (your dataset is ~250KB, so 4K is reasonable; GPT-2 uses 50K for a much larger corpus)
- **Byte-level** (not character-level) so LaTeX math symbols like `∇`, `∑`, `∂` in your scraped data are handled correctly
- **Regex pre-tokenization** to prevent merges across word boundaries (similar to GPT-2's pattern: split on spaces, punctuation, numbers)

---

### Phase 3 — Custom Model Architecture

#### [NEW] [model.py](file:///c:/Users/joshu/OneDrive/projects/llm/model.py)

Replace [gpt.py](file:///c:/Users/joshu/OneDrive/projects/llm/gpt.py) with a modern transformer that uses techniques from LLaMA/Mistral. Each change has a clear "why" you can explain:

##### 3a. RMSNorm (replaces LayerNorm)

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    
    Why: ~15% faster than LayerNorm because it skips the mean-centering step.
    Used by LLaMA, Mistral, Gemma. Same expressiveness in practice.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

##### 3b. Rotary Position Embeddings — RoPE (replaces learned position embeddings)

```python
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """Precompute the complex exponentials for RoPE.
    
    Why: RoPE encodes relative position through rotation of the query/key vectors.
    - No learned parameters (reduces model size)
    - Extrapolates to longer sequences than seen during training
    - Used by every modern open LLM (LLaMA, Mistral, Gemma, Qwen)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64

def apply_rotary_emb(xq, xk, freqs_cis):
    """Apply rotary embeddings to query and key tensors."""
    # Reshape to complex, multiply by rotation, reshape back
```

##### 3c. SwiGLU Feed-Forward (replaces ReLU FFN)

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU activation in feed-forward block (Shazeer, 2020).
    
    Why: Consistently outperforms ReLU/GELU in transformer FFNs.
    The gating mechanism allows the network to selectively pass information.
    Uses 3 linear projections instead of 2, but hidden_dim is scaled
    to 2/3 * 4 * n_embd to keep parameter count equivalent.
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or int(2/3 * 4 * dim)
        # Round to nearest multiple of 8 for GPU efficiency
        hidden_dim = 8 * ((hidden_dim + 7) // 8)
        
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # up projection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

##### 3d. Grouped Query Attention — GQA (replaces standard Multi-Head Attention)

```python
class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (Ainslie et al., 2023).
    
    Why: Standard MHA uses n_head separate K,V projections. GQA shares K,V
    heads across groups of Q heads, reducing KV cache memory by n_head/n_kv_head.
    - n_kv_head = n_head → standard MHA
    - n_kv_head = 1 → Multi-Query Attention
    - n_kv_head = 2-4 → GQA (the sweet spot used by LLaMA 2/3, Mistral)
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_head % config.n_kv_head == 0
        self.n_rep = config.n_head // config.n_kv_head  # repetition factor
        
        self.wq = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.wk = nn.Linear(config.n_embd, config.n_embd // self.n_rep, bias=False)
        self.wv = nn.Linear(config.n_embd, config.n_embd // self.n_rep, bias=False)
        self.wo = nn.Linear(config.n_embd, config.n_embd, bias=False)
```

##### 3e. KV Cache for Inference

```python
class KVCache:
    """Key-Value cache for autoregressive generation.
    
    Why: Without a cache, generating N tokens requires O(N²) compute because
    each step recomputes attention over the entire sequence. With a cache,
    only the new token's K,V are computed — generation becomes O(N).
    This directly improves both standard and speculative decoding speed.
    """
    def __init__(self, batch_size, max_seq_len, n_kv_heads, head_dim, device):
        self.cache_k = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, device=device)
        self.cache_v = torch.zeros(batch_size, n_kv_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0
    
    def update(self, k, v):
        """Append new K,V to cache and return full K,V for attention."""
        ...
    
    def reset(self):
        self.seq_len = 0
```

##### 3f. Full Model Assembly

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.ffn = SwiGLUFFN(config.n_embd, dropout=config.dropout)
    
    def forward(self, x, freqs_cis, kv_cache=None):
        x = x + self.attn(self.attn_norm(x), freqs_cis, kv_cache)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # No position embedding table — RoPE handles positions
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying: output projection shares weights with token embeddings
        self.tok_emb.weight = self.output.weight
        
        # Precompute RoPE frequencies
        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size
        )
    
    def forward(self, idx, targets=None, kv_cache=None):
        """
        Args:
            idx: (B, T) token indices
            targets: (B, T) target indices for training loss, or None for inference
            kv_cache: list of KVCache objects for each layer, or None
        Returns:
            logits: (B, T, vocab_size)
            loss: scalar if targets provided, else None
        """
        ...
    
    def generate(self, idx, max_new_tokens, use_cache=True):
        """Autoregressive generation with optional KV cache."""
        ...
```

> [!IMPORTANT]
> **Weight tying** between the embedding and output layers reduces parameter count by `vocab_size × n_embd` and is standard practice (GPT-2, LLaMA, etc.).

**Architecture comparison (what you're replacing and why):**

| Component | Old (gpt.py) | New (model.py) | Why |
|-----------|-------------|----------------|-----|
| Normalization | LayerNorm | RMSNorm | Faster, same quality |
| Position encoding | Learned embeddings | RoPE | No learned params, length-generalizable |
| FFN activation | ReLU | SwiGLU | Better training dynamics |
| Attention | Standard MHA | GQA | Smaller KV cache, faster inference |
| Inference | Recompute everything | KV Cache | O(N) vs O(N²) generation |
| Linear bias | True | False | Fewer params, modern convention |
| Weight init | Custom | Same (Xavier normal) | Keep what works |

---

### Phase 4 — Training Pipeline

#### [NEW] [train.py](file:///c:/Users/joshu/OneDrive/projects/llm/train.py)

A proper training script with learning rate scheduling, gradient clipping, checkpointing, and experiment tracking.

**Key components:**

1. **Data loading** — read tokenized data, create train/val splits, batch iterator
2. **Cosine learning rate schedule with linear warmup**:
   ```python
   def get_lr(step, config):
       if step < config.warmup_iters:
           return config.learning_rate * step / config.warmup_iters
       decay_ratio = (step - config.warmup_iters) / (config.max_iters - config.warmup_iters)
       coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
       return config.min_lr + coeff * (config.learning_rate - config.min_lr)
   ```
3. **Gradient clipping** — `torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)`
4. **WandB logging** — loss curves, learning rate, gradient norms, generation samples
5. **Checkpointing** — save model state + optimizer state + config + step number
6. **Two training runs**: one for target model (full config), one for draft model (DraftConfig)

```python
# Training loop skeleton
wandb.init(project="custom-llm", config=asdict(train_config))

for step in range(config.max_iters):
    lr = get_lr(step, config)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    wandb.log({"loss": loss.item(), "lr": lr, "grad_norm": grad_norm})
    
    if step % config.eval_interval == 0:
        val_loss = estimate_loss(model, 'val')
        wandb.log({"val_loss": val_loss})
        save_checkpoint(model, optimizer, step, config)
```

---

### Phase 5 — Speculative Decoding Update

#### [MODIFY] [speculative.py](file:///c:/Users/joshu/OneDrive/projects/llm/speculative.py)

The core `speculative_generate` function is already model-agnostic (just refactored). Updates needed:

1. **Remove `import gpt` dependency** — the DraftModel class should use the new `model.py` architecture with `DraftConfig`
2. **Add KV-cache-aware speculative decoding** — the current implementation recomputes the full sequence every step. With KV cache:
   - Draft model uses its own KV cache during proposal
   - Target model processes all draft tokens in one cached pass
   - On rejection, caches are rolled back to the accepted position
3. **Update `__main__` block** to use `model.py` and `tokenizer.py`

> [!TIP]
> KV-cache-aware speculative decoding is a significant engineering challenge and a strong interview talking point. The tricky part is cache rollback on rejection — you need to truncate the cache to `accepted_count` and continue from there.

---

### Phase 6 — Benchmark Update

#### [MODIFY] [benchmark.py](file:///c:/Users/joshu/OneDrive/projects/llm/benchmark.py)

Two benchmark modes:

1. **Custom model benchmark** (primary) — benchmark your own trained target + draft models with speculative decoding
2. **HuggingFace benchmark** (secondary, for comparison) — the existing GPT-2 Large / DistilGPT-2 comparison

Add **matplotlib plots** saved to `results/`:
- Tokens/sec vs gamma (bar chart)
- Acceptance rate vs gamma
- Speedup heatmap (gamma × sequence length)
- KV cache vs no-cache comparison

---

### Phase 7 — Documentation

#### [REWRITE] [README.md](file:///c:/Users/joshu/OneDrive/projects/llm/README.md)

Structure:
1. **Project overview** — one paragraph explaining the end-to-end pipeline
2. **Architecture diagram** (mermaid) showing data flow
3. **Key design decisions** — table of what/why for each component
4. **Results** — benchmark numbers with plots
5. **How to run** — training, generation, benchmarking commands
6. **File-by-file documentation**

---

## Implementation Order

> [!IMPORTANT]
> Each phase builds on the previous one. Don't skip ahead.

| Phase | Files | Depends On | Estimated Effort |
|-------|-------|-----------|-----------------|
| 1. Config | `config.py` | Nothing | ~30 min |
| 2. Tokenizer | `tokenizer.py` | Phase 1 | ~3-4 hours |
| 3. Model | `model.py` | Phase 1 | ~4-5 hours |
| 4. Training | `train.py` | Phases 1-3 | ~2-3 hours |
| 5. Speculative update | `speculative.py` | Phases 1-3 | ~2 hours |
| 6. Benchmark update | `benchmark.py` | Phases 1-5 | ~1-2 hours |
| 7. Documentation | `README.md` | All phases | ~1 hour |

**Total: ~15-20 hours of focused work.**

---

## Open Questions

> [!IMPORTANT]
> **Tokenizer vocab size:** 4096 is a reasonable default for ~250KB of data. Would you prefer a different size? Larger vocab = shorter sequences but sparser embeddings. Smaller vocab = longer sequences but denser learning.

> [!IMPORTANT]
> **Model scale:** The current config (384 dim, 6 heads, 6 layers, ~10M params) trains in minutes on a single GPU. Do you want to keep this scale or go larger? Larger models make the speculative decoding speedup more dramatic, but need more training time and GPU memory.

> [!IMPORTANT]
> **WandB setup:** Do you already have a WandB account? If not, we can use TensorBoard as a local alternative, or set up WandB during Phase 4.

> [!IMPORTANT]
> **Dataset expansion:** Your current corpus is ~250KB (5 Wikipedia articles on ML/AI). Would you like to expand it (more articles, textbooks, arxiv abstracts)? More data = better model, but also longer training and larger tokenizer vocab.

## Verification Plan

### Automated Tests
- **Tokenizer roundtrip**: `assert decode(encode(text)) == text` for diverse inputs (ASCII, Unicode, LaTeX math, empty string, special tokens)
- **Model forward pass**: verify output shapes `(B, T, vocab_size)` for various input sizes
- **KV cache consistency**: verify that cached generation produces identical logits to uncached generation
- **Speculative correctness**: verify that speculative generation output distribution matches standard generation (statistical test over many samples)

### Manual Verification
- **Training curves**: loss decreases smoothly, no NaN/inf, WandB dashboard looks right
- **Generated text quality**: coherent output on ML/AI topics after training
- **Benchmark results**: speculative decoding shows measurable speedup over standard generation
- **Code review**: walk through each file and verify you can explain every line

---

# Phase 2: Tokenizer Implementation Plan

Before I write `tokenizer.py`, I want to align on two critical design decisions for the BPE tokenizer.

## User Review Required

> [!IMPORTANT]
> **1. Vocabulary Size**
> The current plan suggests `vocab_size = 4096`. For a ~250KB dataset, this will result in longer sequences but very dense, well-learned embeddings for each token. Do you want to stick with **4096**, or adjust it (e.g., 1024 for even denser learning, or 8192 for shorter sequences)?

> [!IMPORTANT]
> **2. Regex Pre-tokenization**
> Standard BPE (like GPT-2/LLaMA) splits text into chunks before merging so it doesn't accidentally merge tokens across spaces or punctuation (e.g., "hello" and " world" shouldn't become a single mega-token). 
> I propose using the standard GPT-4 splitting regex: `r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""`. 
> Are you okay with using this industry-standard regex, or would you prefer a simpler one for educational simplicity?

## Proposed Changes

### Tokenizer Component

#### [NEW] [tokenizer.py](file:///c:/Users/joshu/OneDrive/projects/llm/tokenizer.py)
I will implement a `BPETokenizer` class that:
- Uses a byte-level base vocabulary (256 tokens) to handle Unicode/LaTeX seamlessly.
- Applies the regex split to prevent cross-boundary merges.
- Iteratively merges the most frequent pairs until `vocab_size` is reached.
- Adds `<|endoftext|>` and `<|pad|>` as special tokens.
- Supports `save()` and `load()` so we only train the tokenizer once.

## Verification Plan

### Automated Tests
- I will run a round-trip test: `assert decode(encode(text)) == text` on a sample of your dataset containing complex LaTeX math to verify byte-fallback works correctly.
