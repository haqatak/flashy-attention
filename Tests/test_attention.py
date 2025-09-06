import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_flash_attention import FlashAttention
from mlx_flash_attention.attention import (
    flash_attention_forward,
    flash_attention_backward,
    flash_attention_varlen_forward,
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


def standard_attention_4d(q, k, v, scale, mask=None):
    # q, k, v are (B, num_heads, N, head_dim)
    s = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        s = s + mask
    p = mx.softmax(s, axis=-1)
    return p @ v

def test_multi_head_flash_attention():
    B = 1
    N = 256
    num_heads = 4
    head_dim = 32

    q = mx.random.normal((B, N, num_heads, head_dim))
    k = mx.random.normal((B, N, num_heads, head_dim))
    v = mx.random.normal((B, N, num_heads, head_dim))
    scale = 1.0 / np.sqrt(head_dim)

    # Flash attention
    flash_attention = FlashAttention(num_heads, head_dim)
    o_flash = flash_attention(q, k, v)

    # Standard multi-head attention
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    o_standard_4d = standard_attention_4d(q_t, k_t, v_t, scale)
    o_standard_4d = o_standard_4d.transpose(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)

    out_proj_standard = nn.Linear(num_heads * head_dim, num_heads * head_dim)
    out_proj_standard.weight = flash_attention.out_proj.weight
    out_proj_standard.bias = flash_attention.out_proj.bias

    o_standard = out_proj_standard(o_standard_4d)

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
    num_heads = 4
    head_dim = 32
    dropout_p = 0.5

    q = mx.random.normal((B, N, num_heads, head_dim))
    k = mx.random.normal((B, N, num_heads, head_dim))
    v = mx.random.normal((B, N, num_heads, head_dim))

    flash_attention = FlashAttention(num_heads, head_dim, dropout_p=dropout_p)

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
    num_heads = 4
    head_dim = 32
    q_scale = 2.0

    q = mx.random.normal((B, N, num_heads, head_dim))
    k = mx.random.normal((B, N, num_heads, head_dim))
    v = mx.random.normal((B, N, num_heads, head_dim))
    scale = 1.0 / np.sqrt(head_dim)

    # Flash attention
    flash_attention = FlashAttention(num_heads, head_dim)
    o_flash = flash_attention(q, k, v, q_scale=q_scale)

    # Standard multi-head attention with scaled queries
    q_t = q.transpose(0, 2, 1, 3)
    k_t = k.transpose(0, 2, 1, 3)
    v_t = v.transpose(0, 2, 1, 3)

    q_t_scaled = q_t * q_scale

    o_standard_4d = standard_attention_4d(q_t_scaled, k_t, v_t, scale)
    o_standard_4d = o_standard_4d.transpose(0, 2, 1, 3).reshape(B, N, num_heads * head_dim)

    out_proj_standard = nn.Linear(num_heads * head_dim, num_heads * head_dim)
    out_proj_standard.weight = flash_attention.out_proj.weight
    out_proj_standard.bias = flash_attention.out_proj.bias

    o_standard = out_proj_standard(o_standard_4d)

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
    num_heads = 4
    head_dim = 32

    q = mx.random.normal((B, N, num_heads, head_dim))
    k = mx.random.normal((B, N, num_heads, head_dim))
    v = mx.random.normal((B, N, num_heads, head_dim))

    flash_attention = FlashAttention(num_heads, head_dim, dtype=mx.bfloat16)
    o = flash_attention(q, k, v)
    assert o.dtype == mx.bfloat16


def test_flash_attention_varlen():
    B = 2
    N = 256
    M = 256
    D = 64

    q_lens = mx.array([128, 200])
    k_lens = mx.array([100, 256])

    q = mx.random.normal((q_lens.sum().item(), D))
    k = mx.random.normal((k_lens.sum().item(), D))
    v = mx.random.normal((k_lens.sum().item(), D))

    cu_seqlens_q = mx.concatenate([mx.array([0]), mx.cumsum(q_lens)])
    cu_seqlens_k = mx.concatenate([mx.array([0]), mx.cumsum(k_lens)])

    scale = 1.0 / np.sqrt(D)

    # Run flash attention varlen
    o_flash = flash_attention_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, scale)

    # Run standard attention on padded tensors
    q_padded = mx.zeros((B, N, D))
    k_padded = mx.zeros((B, M, D))
    v_padded = mx.zeros((B, M, D))

    for i in range(B):
        q_padded[i, :q_lens[i]] = q[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        k_padded[i, :k_lens[i]] = k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]
        v_padded[i, :k_lens[i]] = v[cu_seqlens_k[i]:cu_seqlens_k[i+1]]

    q_mask = mx.arange(N)[None, :] < q_lens[:, None]
    k_mask = mx.arange(M)[None, :] < k_lens[:, None]
    mask = q_mask[:, :, None] * k_mask[:, None, :]
    mask = mx.where(mask, 0.0, -1e9)

    o_standard, _ = flash_attention_forward(q_padded, k_padded, v_padded, mask=mask, scale=scale)

    # Compare the non-padded parts
    for i in range(B):
        o_flash_i = o_flash[cu_seqlens_q[i]:cu_seqlens_q[i+1]]
        o_standard_i = o_standard[i, :q_lens[i]]
        assert mx.allclose(o_flash_i, o_standard_i, atol=1e-5, rtol=1e-6)

def test_deterministic_mode():
    B = 1
    N = 256
    num_heads = 4
    head_dim = 32
    dropout_p = 0.5

    q = mx.random.normal((B, N, num_heads, head_dim))
    k = mx.random.normal((B, N, num_heads, head_dim))
    v = mx.random.normal((B, N, num_heads, head_dim))

    flash_attention = FlashAttention(num_heads, head_dim, dropout_p=dropout_p)

    # Deterministic
    o1 = flash_attention(q, k, v, deterministic=True)
    o2 = flash_attention(q, k, v, deterministic=True)
    assert mx.allclose(o1, o2)

    # Non-deterministic
    o3 = flash_attention(q, k, v, deterministic=False)
    o4 = flash_attention(q, k, v, deterministic=False)
    assert not mx.allclose(o3, o4)

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
    test_flash_attention_varlen()
    test_deterministic_mode()
