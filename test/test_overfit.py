"""
Overfit Sanity Check
====================
The ultimate proof that a custom transformer architecture works. 
If causal masking, attention, or gradients are broken, it will generate 
incoherent gibberish. If it is mathematically sound, it will perfectly 
memorize a single sequence in ~100 iterations.
"""

import torch
import torch.nn.functional as F
from config import ModelConfig
from model import Transformer

def test_overfit(device):
    # 1. The sequence to memorize
    text = (
        "Hello! If you are reading this exact sentence, it means the model "
        "architecture is mathematically sound. The differential attention works, "
        "the causal masking prevents looking into the future, and the gradients "
        "flow perfectly back through the stages."
    )
    
    # 2. Simple character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    
    # 3. Initialize a small model
    # We use a smaller embedding/layer count so it trains instantly on a single CPU/GPU
    config = ModelConfig(
        vocab_size=vocab_size,
        block_size=data.shape[1],
        n_embd=128,
        n_head=4,
        n_layer=4,
        n_stage=2,
        dropout=0.0,
        exit_weights=(0.3, 0.7)
    )
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train targets are shifted by 1
    idx = data[:, :-1]
    targets = data[:, 1:]
    
    print("Training...")
    model.train()
    for step in range(150):
        optimizer.zero_grad()
        _, loss, _ = model(idx, targets=targets)
        loss.backward()
        optimizer.step()
        
        if step % 25 == 0 or step == 149:
            print(f"  Step {step:3d} | Loss: {loss.item():.4f}")
            
    # 4. Generate from just the first character
    print("\nGenerating from prompt: 'H'")
    model.eval()
    prompt = torch.tensor([[stoi['H']]], dtype=torch.long, device=device)
    
    # Generate the exact length of the remaining sequence
    gen_len = data.shape[1] - 1
    
    with torch.no_grad():
        out_idx = model.generate(prompt, max_new_tokens=gen_len, use_cache=True, temperature=1e-5)
        
    output_text = decode(out_idx[0].tolist())
    
    print("EXPECTED:")
    print(text)
    print("\nACTUAL OUTPUT:")
    print(output_text)
    
    if output_text == text:
        print("\nArchitecture is sound")
    else:
        print("\nOutput is incoherent.")

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_overfit(device)
