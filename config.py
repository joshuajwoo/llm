from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 4096 # Set dynamically after tokenizer training
    block_size: int = 512 # Context window length
    n_embd: int = 768 # Embedding dimension
    n_head: int = 12 # Number of query attention heads
    n_kv_head: int = 4 # KV heads for GQA (n_head must be divisible by this)
    n_layer: int = 12 # Number of transformer blocks
    dropout: float = 0.1
    bias: bool = False # Linear layer bias (modern LLMs skip this)

@dataclass
class DraftConfig(ModelConfig):
    n_embd: int = 64
    n_head: int = 2
    n_kv_head: int = 1
    n_layer: int = 1
    dropout: float = 0.0

@dataclass
class TrainConfig:
    batch_size: int = 64
    max_iters: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5 # Floor for cosine schedule
    warmup_iters: int = 200
    eval_interval: int = 500
    eval_iters: int = 200
    grad_accum_steps: int = 1
    weight_decay: float = 0.1
    grad_clip: float = 1.0

