import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_flash_attention import FlashAttention
from mlx_flash_attention.attention import (
    flash_attention_forward,
    flash_attention_backward,
)


def standard_attention(q, k, v, scale, mask=None):
    """A standard attention implementation."""
    s = (q @ k.transpose(0, 2, 1)) * scale
    if mask is not None:
        s = s + mask
    p = mx.softmax(s, axis=-1)
    return p @ v


def test_flash_attention_forward():
    B = 1
    N = 256
    M = 256
    D = 64
    D_v = 64

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, M, D))
    v = mx.random.normal((B, M, D_v))
    scale = 1.0 / np.sqrt(D)

    # Run flash attention
    o_flash, _ = flash_attention_forward(q, k, v, scale=scale)

    # Run standard attention
    o_standard = standard_attention(q, k, v, scale)

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)


def test_flash_attention_backward():
    B = 1
    N = 128
    M = 128
    D = 32
    D_v = 32

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, M, D))
    v = mx.random.normal((B, M, D_v))
    dO = mx.random.normal((B, N, D_v))
    scale = 1.0 / np.sqrt(D)

    # Standard attention gradients
    standard_attention_grad = mx.grad(
        lambda q, k, v: mx.sum(standard_attention(q, k, v, scale) * dO),
        argnums=(0, 1, 2),
    )
    dQ_standard, dK_standard, dV_standard = standard_attention_grad(q, k, v)

    # Flash attention gradients
    o_flash, L_flash = flash_attention_forward(q, k, v, scale=scale)
    dQ_flash, dK_flash, dV_flash = flash_attention_backward(
        q, k, v, o_flash, L_flash, dO, scale=scale
    )

    # Check that the gradients are close
    assert mx.allclose(dQ_flash, dQ_standard, atol=1e-5, rtol=1e-6)
    assert mx.allclose(dK_flash, dK_standard, atol=1e-5, rtol=1e-6)
    assert mx.allclose(dV_flash, dV_standard, atol=1e-5, rtol=1e-6)


def test_flash_attention_causal():
    B = 1
    N = 256
    D = 64
    D_v = 64

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D_v))
    scale = 1.0 / np.sqrt(D)

    # Run flash attention with causal masking
    o_flash, _ = flash_attention_forward(q, k, v, scale=scale, causal=True)

    # Run standard attention with causal mask
    mask = np.triu(np.full((N, N), -1e9), 1)
    mask = mx.array(mask)
    o_standard = standard_attention(q, k, v, scale, mask=mask)

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)


def test_flash_attention_mask():
    B = 1
    N = 256
    M = 256
    D = 64
    D_v = 64

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, M, D))
    v = mx.random.normal((B, M, D_v))
    scale = 1.0 / np.sqrt(D)
    mask = mx.random.randint(0, 2, (B, N, M)) * -1e9

    # Run flash attention with mask
    o_flash, _ = flash_attention_forward(q, k, v, scale=scale, mask=mask)

    # Run standard attention with mask
    o_standard = standard_attention(q, k, v, scale, mask=mask)

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)


def standard_multi_head_attention(
    q, k, v, d_model, num_heads, scale, q_proj, k_proj, v_proj, out_proj, mask=None, causal=False, q_scale=1.0
):
    d_head = d_model // num_heads
    B, N, D = q.shape
    _, M, _ = k.shape

    # Project and split heads
    q = q_proj(q).reshape(B, N, num_heads, d_head).transpose(0, 2, 1, 3)
    if q_scale != 1.0:
        q = q * q_scale
    k = k_proj(k).reshape(B, M, num_heads, d_head).transpose(0, 2, 1, 3)
    v = v_proj(v).reshape(B, M, num_heads, d_head).transpose(0, 2, 1, 3)

    # Standard attention
    if causal:
        mask = np.triu(np.full((N, N), -1e9), 1)
        mask = mx.array(mask)

    if mask is not None:
        mask = mx.expand_dims(mask, 1)
        mask = mx.broadcast_to(mask, (B, num_heads, N, M))

    output = standard_attention(q, k, v, scale, mask=mask)

    # Reshape and project back
    output = output.transpose(0, 2, 1, 3).reshape(B, N, D)
    return out_proj(output)


def test_multi_head_flash_attention():
    B = 1
    N = 256
    D = 128
    num_heads = 4

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D))
    scale = 1.0 / np.sqrt(D // num_heads)

    # Flash attention
    flash_attention = FlashAttention(D, num_heads)
    o_flash = flash_attention(q, k, v)

    # Standard multi-head attention
    o_standard = standard_multi_head_attention(
        q, k, v, D, num_heads, scale,
        flash_attention.q_proj, flash_attention.k_proj, flash_attention.v_proj, flash_attention.out_proj,
        q_scale=1.0
    )

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)


def test_variable_length_attention():
    B = 2
    N = 256
    M = 256
    D = 64
    D_v = 64

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, M, D))
    v = mx.random.normal((B, M, D_v))
    scale = 1.0 / np.sqrt(D)
    q_lens = mx.array([128, 200])
    k_lens = mx.array([100, 256])

    # Run flash attention
    o_flash, _ = flash_attention_forward(q, k, v, scale=scale, q_lens=q_lens, k_lens=k_lens)

    # Run standard attention with mask
    q_mask = mx.arange(N)[None, :] < q_lens[:, None]
    k_mask = mx.arange(M)[None, :] < k_lens[:, None]
    mask = q_mask[:, :, None] * k_mask[:, None, :]
    mask = mx.where(mask, 0.0, -1e9)
    o_standard = standard_attention(q, k, v, scale, mask=mask)

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)

def test_dropout():
    B = 1
    N = 256
    D = 128
    num_heads = 4
    dropout_p = 0.5

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D))

    flash_attention = FlashAttention(D, num_heads, dropout_p=dropout_p)

    # Eval mode
    flash_attention.eval()
    o_eval = flash_attention(q, k, v)

    # Train mode
    flash_attention.train()
    o_train = flash_attention(q, k, v)

    # Check that the outputs are different
    assert not mx.allclose(o_eval, o_train)

def test_query_scaling():
    B = 1
    N = 256
    D = 128
    num_heads = 4
    q_scale = 2.0

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D))
    scale = 1.0 / np.sqrt(D // num_heads)

    # Flash attention
    flash_attention = FlashAttention(D, num_heads)
    o_flash = flash_attention(q, k, v, q_scale=q_scale)

    # Standard multi-head attention with scaled queries
    o_standard = standard_multi_head_attention(
        q, k, v, D, num_heads, scale,
        flash_attention.q_proj, flash_attention.k_proj, flash_attention.v_proj, flash_attention.out_proj,
        q_scale=q_scale
    )

    # Check that the outputs are close
    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)


def test_sliding_window_attention():
    B = 1
    N = 256
    D = 64
    window_size = (32, 32)

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D))
    scale = 1.0 / np.sqrt(D)

    # Run flash attention
    o_flash, _ = flash_attention_forward(q, k, v, scale=scale, window_size=window_size)

    # Run standard attention with sliding window mask
    query_indices = mx.arange(N).reshape(-1, 1)
    key_indices = mx.arange(N).reshape(1, -1)
    lower_bound_mask = key_indices >= (query_indices - window_size[0])
    upper_bound_mask = key_indices <= (query_indices + window_size[1])
    mask = lower_bound_mask & upper_bound_mask
    mask = mx.where(mask, 0.0, -1e9)
    o_standard = standard_attention(q, k, v, scale, mask=mask)

    assert mx.allclose(o_flash, o_standard, atol=1e-5, rtol=1e-6)

def test_mixed_precision():
    B = 1
    N = 256
    D = 128
    num_heads = 4

    q = mx.random.normal((B, N, D))
    k = mx.random.normal((B, N, D))
    v = mx.random.normal((B, N, D))

    flash_attention = FlashAttention(D, num_heads, dtype=mx.bfloat16)
    o = flash_attention(q, k, v)
    assert o.dtype == mx.bfloat16


if __name__ == "__main__":
    test_flash_attention_forward()
    test_flash_attention_backward()
    test_flash_attention_causal()
    test_flash_attention_mask()
    test_multi_head_flash_attention()
    test_variable_length_attention()
    test_dropout()
    test_query_scaling()
    test_sliding_window_attention()
    test_mixed_precision()
