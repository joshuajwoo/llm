# Speculative decoding (Leviathan et al., 2023)
#
# Draft model generates γ tokens autoregressively: x_i ~ q(x)
# Target model scores all γ+1 positions in one forward pass: p(x)
# Accept token i with probability min(1, p(x_i)/q(x_i))
# Rejection: sample from residual distribution max(0, p - q), normalized
# All accepted: bonus token from target's next-position logits
# Guarantees: output distribution = target distribution exactly

import torch
import torch.nn as nn
from torch.nn import functional as F
import time

import gpt
from draft_model import DraftModel, draft_n_embd, draft_train_steps

max_new_tokens = 200

@torch.no_grad()
def speculative_generate(target_fn, draft_fn, idx, max_new_tokens, gamma=4, block_size=None):
    # target_fn(idx) -> logits (B, T, C), draft_fn(idx) -> logits (B, T, C)
    # idx: (1, prompt_len), gamma: number of draft tokens per round
    total_generated = 0
    total_drafted = 0
    total_accepted = 0

    while total_generated < max_new_tokens:
        current_len = idx.shape[1]
        if block_size and current_len >= block_size - gamma - 1:
            break  # Prevent context window overflow

        # Draft model proposes γ tokens autoregressively
        draft_idx = idx.clone() # (1, current_len)
        draft_probs_list = []
        tokens_to_draft = min(gamma, max_new_tokens - total_generated) # don't overflow generated tokens past max_new_tokens

        for _ in range(tokens_to_draft):
            idx_cond = draft_idx[:, -block_size:] if block_size else draft_idx # clip to context window
            logits = draft_fn(idx_cond) # (1, T, C)
            probs = F.softmax(logits[:, -1, :], dim=-1) # (1, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (1, 1)
            draft_idx = torch.cat((draft_idx, idx_next), dim=1) # (1, T+1)
            draft_probs_list.append(probs)

        draft_tokens = draft_idx[:, current_len:] # (1, γ)

        # Target model verifies all γ drafts in ONE forward pass
        target_input = draft_idx[:, -block_size:] if block_size else draft_idx
        target_logits = target_fn(target_input) # (1, T, C)
        eval_slice = target_logits[:, -(tokens_to_draft + 1):-1, :] # (1, γ, C) logits of target model, picking logit probability BEFORE each draft position
        target_probs_list = [F.softmax(eval_slice[:, i, :], dim=-1)
                             for i in range(tokens_to_draft)]

        # Stochastic acceptance — preserves exact target distribution
        accepted_count = 0
        for i in range(tokens_to_draft):
            sampled_token = draft_tokens[0, i].item() # scalar int
            p_target = target_probs_list[i][0, sampled_token].item() # scalar float from (1, C)
            p_draft = draft_probs_list[i][0, sampled_token].item() # scalar float from (1, C)

            # if p/q >= 1 (target agrees), rand always passes. if p/q < 1, accept proportionally
            if torch.rand(1).item() < p_target / p_draft: # torch.rand(1) is (1,)
                accepted_count += 1
            else:
                break # reject this and all subsequent tokens

        total_drafted += tokens_to_draft
        total_accepted += accepted_count

        # Append accepted tokens + sample one more
        if accepted_count > 0:
            idx = torch.cat((idx, draft_tokens[:, :accepted_count]), dim=1) # keep accepted drafts
            total_generated += accepted_count

        if accepted_count < tokens_to_draft:
            # sample from residual distribution norm(max(0, p - q))
            p_t = target_probs_list[accepted_count] # (1, C)
            p_d = draft_probs_list[accepted_count] # (1, C)
            correction_dist = torch.clamp(p_t - p_d, min=0.0) # (1, C)
            correction_sum = correction_dist.sum(dim=-1, keepdim=True)
            if correction_sum.item() > 0:
                correction_dist = correction_dist / correction_sum # normalize
            else:
                correction_dist = p_t # fallback to target
            next_token = torch.multinomial(correction_dist, num_samples=1) # (1, 1)
        else:
            # all γ drafts accepted → free extra token from target
            bonus_probs = F.softmax(target_logits[:, -1, :], dim=-1) # (1, C)
            next_token = torch.multinomial(bonus_probs, num_samples=1) # (1, 1)

        idx = torch.cat((idx, next_token), dim=1) # at least 1 token/round
        total_generated += 1

    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0.0
    return idx, acceptance_rate  # (1, prompt_len + generated), scalar


# EXECUTION LOGIC
if __name__ == '__main__':
    print("Initializing Target Model from gpt.py...")
    target_model = gpt.GPTLanguageModel().to(gpt.device)

    print("Initializing fast Draft Model...")
    draft_model = DraftModel().to(gpt.device)

    # Train draft model briefly — just enough to correlate with target
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

    # Wrap models: speculative_generate expects callable(idx) -> logits (B, T, V)
    target_fn = lambda idx: target_model(idx)[0]
    draft_fn = lambda idx: draft_model(idx)[0]

    print("\nStandard Generation Profiling")
    t0 = time.time()
    _ = target_model.generate(context.clone(), max_new_tokens=max_new_tokens)
    std_time = time.time() - t0
    print(f"Standard Time: {std_time:.4f} seconds")

    print("\nSpeculative Generation Profiling")
    t1 = time.time()
    _, acc = speculative_generate(target_fn, draft_fn, context.clone(),
                                  max_new_tokens=max_new_tokens,
                                  block_size=gpt.block_size)
    spec_time = time.time() - t1
    print(f"Speculative Time: {spec_time:.4f} seconds")
    print(f"Acceptance Rate: {acc:.1%}")

    print(f"\nOptimization Factor: {std_time / spec_time:.2f}x Speedup")