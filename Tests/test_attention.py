import mlx.core as mx
import numpy as np
from mlx_flash_attention.attention import (
    flash_attention_forward,
    flash_attention_backward,
)


def standard_attention(q, k, v, scale):
    """A standard attention implementation."""
    s = (q @ k.transpose(0, 2, 1)) * scale
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


if __name__ == "__main__":
    test_flash_attention_forward()
    test_flash_attention_backward()
