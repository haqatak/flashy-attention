import mlx.core as mx
import mlx.nn as nn
from .attention import flash_attention_forward
from typing import Tuple


class FlashAttention(nn.Module):
    """
    An implementation of multi-head flash attention as a `mlx.nn.Module`.

    This module expects the queries, keys, and values to be already projected
    and split into heads.
    """

    def __init__(self, num_heads: int, head_dim: int, dropout_p: float = 0.0, dtype=mx.float32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype

        self.out_proj = nn.Linear(num_heads * head_dim, num_heads * head_dim)
        self.dropout = nn.Dropout(p=dropout_p)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array = None,
        causal: bool = False,
        q_lens: mx.array = None,
        k_lens: mx.array = None,
        q_scale: float = 1.0,
        window_size: Tuple[int, int] = (-1, -1),
        deterministic: bool = False,
    ) -> mx.array:
        """
        Args:
            queries (mx.array): The query vectors, in B x N x num_heads x head_dim.
            keys (mx.array): The key vectors, in B x M x num_heads x head_dim.
            values (mx.array): The value vectors, in B x M x num_heads x head_dim.
            mask (mx.array, optional): The attention mask. Defaults to None.
            causal (bool, optional): If True, apply a causal mask. Defaults to False.
            q_lens (mx.array, optional): The sequence lengths of the queries, in B.
            k_lens (mx.array, optional): The sequence lengths of the keys, in B.
            q_scale (float, optional): The scale factor for the queries. Defaults to 1.0.
            window_size (Tuple[int, int], optional): The size of the sliding window.
                Defaults to (-1, -1), which means no sliding window.
            deterministic (bool, optional): If True, use a fixed seed for dropout.
                Defaults to False.

        Returns:
            mx.array: The output vectors, in B x N x (num_heads * head_dim).
        """
        if deterministic:
            mx.random.seed(0)

        B, N, _, _ = queries.shape
        _, M, _, _ = keys.shape

        queries = queries.astype(self.dtype)
        keys = keys.astype(self.dtype)
        values = values.astype(self.dtype)

        # Transpose to (B, num_heads, N, head_dim)
        q = queries.transpose(0, 2, 1, 3)
        k = keys.transpose(0, 2, 1, 3)
        v = values.transpose(0, 2, 1, 3)

        # Scale queries
        if q_scale != 1.0:
            q = q * q_scale

        # Reshape for flash attention
        q = q.reshape(B * self.num_heads, N, self.head_dim)
        k = k.reshape(B * self.num_heads, M, self.head_dim)
        v = v.reshape(B * self.num_heads, M, self.head_dim)

        if q_lens is not None:
            q_lens = mx.expand_dims(q_lens, 1)
            q_lens = mx.broadcast_to(q_lens, (B, self.num_heads))
            q_lens = q_lens.reshape(B * self.num_heads)

        if k_lens is not None:
            k_lens = mx.expand_dims(k_lens, 1)
            k_lens = mx.broadcast_to(k_lens, (B, self.num_heads))
            k_lens = k_lens.reshape(B * self.num_heads)

        if mask is not None:
            mask = mx.expand_dims(mask, 1)
            mask = mx.broadcast_to(mask, (B, self.num_heads, N, M))
            mask = mask.reshape(B * self.num_heads, N, M)

        # Flash attention
        output, _ = flash_attention_forward(
            q,
            k,
            v,
            mask=mask,
            scale=1.0 / q.shape[-1] ** 0.5,
            causal=causal,
            q_lens=q_lens,
            k_lens=k_lens,
            window_size=window_size,
        )

        # Reshape and project back
        output = output.reshape(B, self.num_heads, N, self.head_dim).transpose(0, 2, 1, 3)
        output = output.reshape(B, N, self.num_heads * self.head_dim)

        # Apply output projection and dropout. Dropout is applied to the final output
        # of the attention layer, which is a common practice.
        return self.dropout(self.out_proj(output))
