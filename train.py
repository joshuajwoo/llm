"""
Training Loop
=============
Trains the Waypoint Transformer on the binary dataset (`train.pt`).
Uses gradient accumulation to simulate large batch sizes on consumer hardware.
Saves checkpoints periodically.
"""

import os
import time
import math
import torch
from config import ModelConfig, TrainConfig
from model import Transformer

def get_batch(data, config, train_cfg, device):
    ix = torch.randint(len(data) - config.block_size, (train_cfg.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+config.block_size] for i in ix])
    return x.long().to(device), y.long().to(device)

def get_lr(it, train_cfg):
    # Linear warmup
    if it < train_cfg.warmup_iters:
        return train_cfg.learning_rate * it / train_cfg.warmup_iters
    # Cosine decay
    if it > train_cfg.max_iters:
        return train_cfg.min_lr
    decay_ratio = (it - train_cfg.warmup_iters) / (train_cfg.max_iters - train_cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_cfg.min_lr + coeff * (train_cfg.learning_rate - train_cfg.min_lr)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device.upper()}")
    
    cfg = ModelConfig()
    train_cfg = TrainConfig()
    
    # Paths
    train_path = "data/train.pt"
    ckpt_path = "data/checkpoint.pt"
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"{train_path} doesn't exist.")
        
    print("Loading data")
    data = torch.load(train_path, weights_only=True)
    print(f"Dataset size: {len(data):,} tokens")
    
    model = Transformer(cfg).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg.learning_rate, 
        weight_decay=train_cfg.weight_decay
    )
    start_iter = 0
    if os.path.exists(ckpt_path):
        print(f"Resuming training from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter_num']
        print(f"Resumed at iteration {start_iter}")
        
    model.train()
    
    t0 = time.time()
    try:
        for iter_num in range(start_iter, train_cfg.max_iters):
            lr = get_lr(iter_num, train_cfg)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            optimizer.zero_grad(set_to_none=True)
            
            # Gradient Accumulation
            accum_loss = 0.0
            for micro_step in range(train_cfg.grad_accum_steps):
                x, y = get_batch(data, cfg, train_cfg, device)
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    _, loss, exit_info = model(x, targets=y)
                loss = loss / train_cfg.grad_accum_steps
                
                accum_loss += loss.item()
                loss.backward()
                
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            
            # Logging
            if True:
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                # Calculate tokens per second
                tokens_per_iter = train_cfg.grad_accum_steps * train_cfg.batch_size * cfg.block_size
                tps = (tokens_per_iter) / dt if iter_num > 0 else 0.0
                
                print(f"Iter {iter_num:4d} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | Tok/sec: {tps:.1f}")
                
                # Print exit statistics occasionally
                if iter_num % 50 == 0:
                    print("  Exit Confidences: " + ", ".join([f"Stage {info['stage']}: {info['confidence'].mean().item():.2f}" for info in exit_info if info['stage'] < cfg.n_stage - 1]))
                    
            # Checkpointing
            if iter_num > 0 and iter_num % train_cfg.eval_interval == 0:
                print(f"Saving checkpoint to {ckpt_path}...")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'config': cfg,
                    'train_config': train_cfg
                }
                torch.save(checkpoint, ckpt_path)
                
    except KeyboardInterrupt:
        pass
    
    print(f"\nSaving checkpoint to {ckpt_path}")
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter_num': iter_num if 'iter_num' in locals() else 0,
        'config': cfg,
        'train_config': train_cfg
    }
    torch.save(checkpoint, ckpt_path)
    
if __name__ == '__main__':
    main()
