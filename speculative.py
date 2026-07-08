"""
Speculative Decoding Demonstration
==================================
This script demonstrates "Self-Speculative Decoding".
Instead of using a separate small draft model, we use our own model's 
Early Exits (Stage 0) as the "draft" model, and the Final Stage as the "verifier".

This allows us to generate multiple tokens in a single forward pass!
"""

import time
import torch
import torch.nn.functional as F
from config import ModelConfig
from model import Transformer
from tokenizer import BPETokenizer

def load_model(device, ckpt_path="data/checkpoint.pt"):
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = checkpoint.get('config', ModelConfig())
    model = Transformer(cfg).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg

def speculative_generate(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # We will track how many tokens were successfully drafted (accepted)
    accepted_drafts = 0
    total_steps = 0
    
    t0 = time.perf_counter()
    
    with torch.no_grad():
        while idx.shape[1] < max_new_tokens:
            total_steps += 1
            
            # --- STEP 1: The Draft (Fast) ---
            # We run just the very first stage (Stage 0) to quickly guess the next token
            draft_logits, _, _ = model(idx)
            # draft_logits contains the output from the final stage by default, 
            # but to simulate a true draft, we can use the model's exit_heads.
            # For simplicity in this demo, we'll just simulate getting a draft token.
            # In a full implementation, we'd slice the model to only run Stage 0.
            
            # We'll extract the Stage 0 logits directly using a partial forward pass:
            x = model.tok_emb(idx)
            sinks = model.sink_tokens.expand(1, -1, -1)
            x = torch.cat([sinks, x], dim=1)
            freqs = model.freqs_cis[:x.shape[1]]
            
            # Run only Stage 0
            for layer_idx in range(model.layers_per_stage):
                x = model.blocks[layer_idx](x, freqs, None)
                
            h_real = x[:, model.config.n_sinks:, :]
            draft_logits, _, _ = model.exit_heads[0](h_real)
            
            draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
            
            # --- STEP 2: The Verification (Parallel) ---
            # We append the draft token to our sequence, and run the FULL model 
            # on the combined sequence to check if the draft was correct.
            idx_with_draft = torch.cat([idx, draft_token], dim=1)
            
            full_logits, _, _ = model(idx_with_draft)
            
            # The full model tells us what the TRUE next token should have been
            true_next_token = torch.argmax(full_logits[:, -2, :], dim=-1, keepdim=True)
            
            if true_next_token.item() == draft_token.item():
                # SUCCESS! The draft was correct. 
                # We get to keep the draft token AND the token after it!
                accepted_drafts += 1
                next_next_token = torch.argmax(full_logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat([idx, draft_token, next_next_token], dim=1)
                
                # Print the two tokens we just got in one step
                new_text = tokenizer.decode([draft_token.item(), next_next_token.item()])
                print(new_text, end="", flush=True)
            else:
                # REJECTED! The draft was wrong.
                # We throw away the draft token, and just use the true token.
                idx = torch.cat([idx, true_next_token], dim=1)
                
                new_text = tokenizer.decode([true_next_token.item()])
                print(new_text, end="", flush=True)
                
    t1 = time.perf_counter()
    dt = t1 - t0
    
    print("\n" + "-" * 40)
    print(f"Time: {dt:.2f}s | Speed: {(idx.shape[1] / dt):.1f} tokens/sec")
    print(f"Speculative Drafts Accepted: {accepted_drafts} (Tokens generated for 'free')")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, cfg = load_model(device)
    except FileNotFoundError as e:
        print(e)
        return
        
    tokenizer = BPETokenizer(vocab_size=cfg.vocab_size)
    tokenizer.load("data/tokenizer")
    
    prompt = input("\nEnter a prompt for Speculative Decoding: ")
    speculative_generate(model, tokenizer, prompt, max_new_tokens=50, device=device)

if __name__ == '__main__':
    main()