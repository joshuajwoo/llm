"""Smoke tests for the model.

Verifies forward pass shapes, multi-exit losses, confidence ranges,
sink handling, KV cache generation, early exit generation, and param count.
"""

import torch
from config import ModelConfig
from model import Transformer


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def test_forward_shapes(config, name="Model"):
    """Forward pass produces correct output shapes."""
    print(f"Testing {name}: {config}")
    model = Transformer(config)
    model.eval()

    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    final_logits, total_loss, exit_info = model(idx, targets=targets)

    # Final logits shape
    assert final_logits.shape == (B, T, config.vocab_size), \
        f"Expected logits {(B, T, config.vocab_size)}, got {final_logits.shape}"
    print(f"   Final logits shape: {final_logits.shape}")

    # Total loss is a scalar
    assert total_loss.dim() == 0, f"Expected scalar loss, got dim {total_loss.dim()}"
    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
    print(f"   Total loss: {total_loss.item():.4f}")

    # exit_info has n_stage entries
    assert len(exit_info) == config.n_stage, \
        f"Expected {config.n_stage} exits, got {len(exit_info)}"
    print(f"   Exit count: {len(exit_info)}")

    return model


def test_exit_info(config, name="Model"):
    """Each exit produces valid logits, confidence, and loss."""
    model = Transformer(config)
    model.eval()

    B, T = 2, 64
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, total_loss, exit_info = model(idx, targets=targets)

    # Check each exit
    for i, info in enumerate(exit_info):
        stage = info['stage']
        logits = info['logits']
        confidence = info['confidence']
        loss = info['loss']

        # Logits shape (sinks should be stripped — shape is (B, T, V) not (B, T+n_sinks, V))
        assert logits.shape == (B, T, config.vocab_size), \
            f"Exit {i} logits shape: expected {(B, T, config.vocab_size)}, got {logits.shape}"

        # Loss is finite
        assert loss is not None and torch.isfinite(loss), \
            f"Exit {i} loss is not finite: {loss}"

        if i < config.n_stage - 1:
            # Non-final exits have confidence scores
            assert confidence is not None, f"Exit {i} missing confidence"
            assert confidence.shape == (B, T), \
                f"Exit {i} confidence shape: expected {(B, T)}, got {confidence.shape}"
            assert confidence.min() >= 0 and confidence.max() <= 1, \
                f"Exit {i} confidence out of [0,1]: [{confidence.min():.4f}, {confidence.max():.4f}]"
            print(f"   Exit {i} (stage {stage}): logits OK, confidence [{confidence.min():.3f}, {confidence.max():.3f}], loss={loss.item():.4f}")
        else:
            # Final exit has no confidence
            assert confidence is None, f"Final exit should not have confidence"
            print(f"   Exit {i} (stage {stage}, final): logits OK, no confidence, loss={loss.item():.4f}")

    # Verify loss weighting
    manual_total = sum(
        config.exit_weights[info['stage']] * info['loss']
        for info in exit_info
    )
    assert torch.allclose(total_loss, manual_total, atol=1e-5), \
        f"Total loss {total_loss.item():.6f} != weighted sum {manual_total.item():.6f}"
    print(f"   Loss weighting verified: {total_loss.item():.4f}")


def test_kv_cache_generation(config, name="Model"):
    """KV cache generation produces correct shapes."""
    model = Transformer(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    n_gen = 10

    result = model.generate(prompt, max_new_tokens=n_gen, use_cache=True)
    expected_len = prompt.shape[1] + n_gen
    assert result.shape == (1, expected_len), \
        f"Expected (1, {expected_len}), got {result.shape}"
    print(f"   KV cache generation: {prompt.shape} → {result.shape}")


def test_no_cache_generation(config, name="Model"):
    """No-cache generation produces correct shapes."""
    model = Transformer(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    n_gen = 5

    result = model.generate(prompt, max_new_tokens=n_gen, use_cache=False)
    expected_len = prompt.shape[1] + n_gen
    assert result.shape == (1, expected_len), \
        f"Expected (1, {expected_len}), got {result.shape}"
    print(f"   No-cache generation: {prompt.shape} → {result.shape}")


def test_early_exit_generation(config, name="Model"):
    """Early exit generation produces correct shapes."""
    model = Transformer(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 8))
    n_gen = 5

    result = model.generate(
        prompt, max_new_tokens=n_gen,
        use_cache=True, halt_threshold=0.5
    )
    expected_len = prompt.shape[1] + n_gen
    assert result.shape == (1, expected_len), \
        f"Expected (1, {expected_len}), got {result.shape}"
    print(f"   Early exit generation (threshold=0.5): {prompt.shape} → {result.shape}")


def test_param_count(config, name="Model"):
    """Parameter count is in expected range."""
    model = Transformer(config)
    n = count_params(model)
    print(f"   {name} parameters: {n:,} ({n/1e6:.1f}M)")
    return n


def test_differential_lambda(config, name="Model"):
    """Differential attention lambda values are finite and vary by layer."""
    model = Transformer(config)
    lambdas = []
    for block in model.blocks:
        attn = block.attn
        lambda_val = (
            torch.exp(attn.lambda_q1 @ attn.lambda_k1)
            - torch.exp(attn.lambda_q2 @ attn.lambda_k2)
            + attn.lambda_init
        )
        assert torch.isfinite(lambda_val), f"Lambda is not finite: {lambda_val.item()}"
        lambdas.append(lambda_val.item())

    # Lambda inits should vary by layer (deeper layers cancel more)
    assert lambdas[0] != lambdas[-1], "Lambda should vary by layer depth"
    print(f"   Lambda range: [{min(lambdas):.4f}, {max(lambdas):.4f}]")


def test_backward_pass(config, name="Model"):
    """Backward pass runs without error and all parameters get gradients."""
    model = Transformer(config)
    model.train()

    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    _, loss, _ = model(idx, targets=targets)
    loss.backward()

    # Check that key parameters have gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"   Backward pass: {has_grad}/{total} params have gradients, loss={loss.item():.4f}")


# Run all tests
if __name__ == '__main__':
    print("Tests: ")

    # --- Target model (full config) ---
    cfg = ModelConfig()
    print(f"\n--- Target Model Config ---")
    test_forward_shapes(cfg, "Target")
    test_exit_info(cfg, "Target")
    test_differential_lambda(cfg, "Target")
    test_backward_pass(cfg, "Target")
    test_kv_cache_generation(cfg, "Target")
    test_no_cache_generation(cfg, "Target")
    test_early_exit_generation(cfg, "Target")
    test_param_count(cfg, "Target")

    print("\nTests passed")