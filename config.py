from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    vocab_size: int = 4096      # Set dynamically after tokenizer training
    block_size: int = 1024      # Context window (drop to 512 for small corpus)
    n_embd: int = 768           # Embedding dimension
    n_head: int = 12            # Differential attention heads
    n_layer: int = 12           # Total transformer layers
    n_stage: int = 4            # Stages (n_layer must be divisible by n_stage)
    dropout: float = 0.1
    bias: bool = False          # Linear layer bias (modern LLMs skip this)
    # Hybrid attention
    window_size: int = 256      # Sliding window for local-attention layers
    n_sinks: int = 4            # Learned sink tokens prepended to every sequence
    # Differential attention
    lambda_init: float = 0.8    # Base lambda for differential attention
    # Early exit
    exit_hidden: int = 128      # Hidden dim for halting MLP (n_embd -> exit_hidden -> 1)
    # Multi-token prediction
    mtp_weight: float = 0.2     # Weight for t+2 auxiliary loss at each exit
    # Exit loss weights per stage (must sum to ~1.0)
    exit_weights: tuple = (0.15, 0.15, 0.2, 0.5)

@dataclass
class TrainConfig:
    batch_size: int = 4      # Micro-batch (use grad accum for effective ~100K tokens/step)
    max_iters: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5        # Floor for cosine schedule
    warmup_iters: int = 200
    eval_interval: int = 500
    eval_iters: int = 200
    grad_accum_steps: int = 16  # Effective batch = 1 * 8 * 1024 ~ 8K tokens
    weight_decay: float = 0.1
    grad_clip: float = 1.0
