import torch
import torch.nn as nn
from torch.nn import functional as F
import time

import gpt

draft_n_embd = 64
draft_train_steps = 500
gamma = 4 # speculation depth (tokens drafted per round)
max_new_tokens = 200

# A miniature, 1-layer, draft_n_embd-dimension model designed strictly for fast guessing.
class DraftModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(gpt.vocab_size, draft_n_embd)
        self.position_embedding_table = nn.Embedding(gpt.block_size, draft_n_embd)
        
        # A single, tiny attention head
        self.key = nn.Linear(draft_n_embd, draft_n_embd, bias=False)
        self.query = nn.Linear(draft_n_embd, draft_n_embd, bias=False)
        self.value = nn.Linear(draft_n_embd, draft_n_embd, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(gpt.block_size, gpt.block_size)))
        
        self.lm_head = nn.Linear(draft_n_embd, gpt.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=gpt.device))
        x = tok_emb + pos_emb
        
        # Fast Attention
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * (draft_n_embd**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        x = wei @ v
        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

# --- 2. SPECULATIVE GENERATION ENGINE ---
@torch.no_grad()
def speculative_generate(target_model, draft_model, idx, max_new_tokens, gamma=gamma):
    target_model.eval()
    draft_model.eval()
    
    total_generated = 0
    
    while total_generated < max_new_tokens:
        current_len = idx.shape[1]
        if current_len >= gpt.block_size - gamma - 1:
            break # Prevent context window overflow
            
        # STEP 1: Draft Model guesses 'gamma' tokens sequentially
        draft_idx = idx.clone() # (1, current_len)
        draft_probs_list = []
        
        for _ in range(gamma):
            idx_cond = draft_idx[:, -gpt.block_size:] # (1, min(current_len+k, block_size))
            logits, _ = draft_model(idx_cond) # (1, T_cond, vocab_size)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (1, vocab_size)
            
            idx_next = torch.multinomial(probs, num_samples=1) # (1, 1)
            draft_idx = torch.cat((draft_idx, idx_next), dim=1) # (1, current_len + k + 1)
            draft_probs_list.append(probs) # list of gamma × (1, vocab_size)
            
        draft_tokens = draft_idx[:, current_len:] # (1, gamma)
        
        # STEP 2: Target Model verifies all guesses in one forward pass
        target_input = draft_idx[:, -gpt.block_size:] # (1, min(current_len+gamma, block_size))
        target_logits, _ = target_model(target_input) # (1, T_input, vocab_size)
        eval_slice = target_logits[:, -(gamma + 1):-1, :] # (1, gamma, vocab_size)
        target_probs_list = [F.softmax(eval_slice[:, i, :], dim=-1) # list of gamma × (1, vocab_size)
                             for i in range(gamma)]
        
        # STEP 3: Stochastic Acceptance Check
        accepted_count = 0
        for i in range(gamma):
            sampled_token = draft_tokens[0, i].item()
            p_target = target_probs_list[i][0, sampled_token].item()
            p_draft = draft_probs_list[i][0, sampled_token].item()
            
            if torch.rand(1).item() < p_target / p_draft:
                accepted_count += 1
            else:
                break 
                
        # STEP 4: Append accepted tokens and handle rejection / bonus sampling
        if accepted_count > 0:
            idx = torch.cat((idx, draft_tokens[:, :accepted_count]), dim=1) # (1, current_len + accepted_count)
            total_generated += accepted_count
            
        if accepted_count < gamma:
            # Rejection path: sample from the residual distribution
            rejected_pos = accepted_count
            p_t = target_probs_list[rejected_pos] # (1, vocab_size)
            p_d = draft_probs_list[rejected_pos] # (1, vocab_size)
            
            correction_dist = torch.clamp(p_t - p_d, min=0.0) # (1, vocab_size)
            correction_dist /= correction_dist.sum(dim=-1, keepdim=True) # (1, vocab_size)
            next_token = torch.multinomial(correction_dist, num_samples=1) # (1, 1)
        else:
            # Bonus path: all gamma drafts accepted, grab one free extra token
            bonus_probs = F.softmax(target_logits[:, -1, :], dim=-1) # (1, vocab_size)
            next_token = torch.multinomial(bonus_probs, num_samples=1) # (1, 1)
            
        idx = torch.cat((idx, next_token), dim=1) # (1, current_len + accepted_count + 1)
        total_generated += 1
        
    return idx

# --- 3. EXECUTION LOGIC ---
if __name__ == '__main__':
    print("Initializing Target Model from gpt.py...")
    target_model = gpt.GPTLanguageModel().to(gpt.device)
    
    print("Initializing fast Draft Model...")
    draft_model = DraftModel().to(gpt.device)
    
    # We train the draft model for a few steps just so it doesn't spit out complete noise
    print(f"Training Draft Model for {draft_train_steps} steps...")
    opt_draft = torch.optim.AdamW(draft_model.parameters(), lr=gpt.learning_rate)
    for i in range(draft_train_steps):
        xb, yb = gpt.get_batch('train')
        _, loss = draft_model(xb, yb)
        opt_draft.zero_grad(set_to_none=True)
        loss.backward()
        opt_draft.step()
    print("Draft Model trained.")

    context = torch.zeros((1, 1), dtype=torch.long, device=gpt.device)

    # Note: Since the Target Model in this script is initialized randomly and not 
    # fully trained in this run, the text output will be gibberish. 
    # This test strictly measures the hardware execution speed.
    
    print("\n--- Standard Generation Profiling ---")
    t0 = time.time()
    _ = target_model.generate(context.clone(), max_new_tokens=max_new_tokens)
    std_time = time.time() - t0
    print(f"Standard Time: {std_time:.4f} seconds")

    print("\n--- Speculative Generation Profiling ---")
    t1 = time.time()
    _ = speculative_generate(target_model, draft_model, context.clone(), max_new_tokens=max_new_tokens)
    spec_time = time.time() - t1
    print(f"Speculative Time: {spec_time:.4f} seconds")

    print(f"\nOptimization Factor: {std_time / spec_time:.2f}x Speedup")