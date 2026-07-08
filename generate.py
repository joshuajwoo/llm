"""
Generation Script
=================
Loads a trained Waypoint model and generates text interactively.
Allows toggling the `halt_threshold` to see the early exit speedup.
"""

import os
import time
import torch
from config import ModelConfig
from model import Transformer
from tokenizer import BPETokenizer

def load_model(device, ckpt_path="data/checkpoint.pt"):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found. Train the model first.")
        
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    cfg = checkpoint.get('config', ModelConfig())
    model = Transformer(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    return model, cfg

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Generating on: {device.upper()}")
    
    try:
        model, cfg = load_model(device)
    except FileNotFoundError as e:
        print(e)
        return
        
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tokenizer.load("data/tokenizer")
    
    print("\n--- Waypoint Interactive Generator ---")
    print("Type 'quit' to exit.")
    
    while True:
        prompt_text = input("\nPrompt: ")
        if prompt_text.lower() == 'quit':
            break
            
        threshold_str = input("Halt Threshold (e.g., 0.8) [Leave blank for full depth]: ")
        if threshold_str.strip() == "":
            halt_threshold = None
        else:
            try:
                halt_threshold = float(threshold_str)
            except ValueError:
                print("Invalid threshold. Using full depth.")
                halt_threshold = None
                
        print("\n" + "-"*40)
        print(prompt_text, end="")
        
        # Encode prompt
        idx = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long, device=device)
        
        # We need a small loop here to decode tokens as they stream, 
        # but model.generate returns the full sequence.
        # For simplicity, we just generate everything and then decode.
        gen_len = 1000
        
        t0 = time.perf_counter()
        with torch.no_grad():
            out_idx = model.generate(
                idx, 
                max_new_tokens=gen_len, 
                temperature=0.8, 
                use_cache=True, 
                halt_threshold=halt_threshold
            )
        t1 = time.perf_counter()
        
        # Decode only the newly generated tokens
        new_ids = out_idx[0, idx.shape[1]:].tolist()
        generated_text = tokenizer.decode(new_ids)
        
        print(generated_text)
        print("-" * 40)
        
        dt = t1 - t0
        tps = gen_len / dt
        depth_msg = f"Threshold: {halt_threshold}" if halt_threshold is not None else "Full Depth (No Early Exit)"
        print(f"[{depth_msg} | Time: {dt:.2f}s | Speed: {tps:.1f} tokens/sec]")

if __name__ == '__main__':
    main()