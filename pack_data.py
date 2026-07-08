"""
Data Packer
===========
Reads the raw text dataset, tokenizes it using our custom BPE tokenizer,
and saves it as highly efficient PyTorch tensors (`train.pt` and `val.pt`).
This prevents the training loop from having to tokenize gigabytes of text on the fly.
"""

import os
import torch
import numpy as np
from tokenizer import BPETokenizer
from config import ModelConfig

def pack_data():
    cfg = ModelConfig()
    
    # Paths
    input_path = "data/input.txt"
    tokenizer_prefix = "data/tokenizer"
    train_path = "data/train.pt"
    val_path = "data/val.pt"
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found. Please run scrape.py first.")
    
    if not os.path.exists(tokenizer_prefix + ".merges"):
        print("Tokenizer not found. Training tokenizer on input text first...")
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
        tokenizer.train(text, verbose=True)
        tokenizer.save(tokenizer_prefix)
    else:
        print("Loading existing tokenizer...")
        tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
        tokenizer.load(tokenizer_prefix)
        
    print("\nReading text dataset...")
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    print(f"Tokenizing {len(text):,} characters (this may take a minute)...")
    ids = tokenizer.encode(text)
    print(f"Total tokens: {len(ids):,}")
    
    # Split 90% train, 10% val
    n = len(ids)
    train_ids = ids[:int(n*0.9)]
    val_ids = ids[int(n*0.9):]
    
    # Convert to 16-bit integers to save memory (vocab < 65536)
    dtype = torch.int16 if cfg.vocab_size < 32768 else torch.int32
    
    print("\nSaving binary tensors...")
    train_tensor = torch.tensor(train_ids, dtype=dtype)
    val_tensor = torch.tensor(val_ids, dtype=dtype)
    
    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)
    
    print(f"Saved {train_path} ({len(train_tensor):,} tokens)")
    print(f"Saved {val_path} ({len(val_tensor):,} tokens)")
    print("Done! Ready for train.py.")

if __name__ == '__main__':
    pack_data()
