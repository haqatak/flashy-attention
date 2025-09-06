import mlx.core as mx
import mlx.nn as nn
from .attention import flash_attention_forward


class FlashAttention(nn.Module):
    """
    An implementation of multi-head flash attention as a `mlx.nn.Module`.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array = None,
        causal: bool = False,
    ) -> mx.array:
        """
        Args:
            queries (mx.array): The query vectors, in B x N x D.
            keys (mx.array): The key vectors, in B x M x D.
            values (mx.array): The value vectors, in B x M x D_v.
            mask (mx.array, optional): The attention mask. Defaults to None.
            causal (bool, optional): If True, apply a causal mask. Defaults to False.

        Returns:
            mx.array: The output vectors.
        """
        B, N, D = queries.shape
        _, M, _ = keys.shape

        # Project and split heads
        q = self.q_proj(queries).reshape(B, N, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(keys).reshape(B, M, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(values).reshape(B, M, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        # Reshape for flash attention
        q = q.reshape(B * self.num_heads, N, self.d_head)
        k = k.reshape(B * self.num_heads, M, self.d_head)
        v = v.reshape(B * self.num_heads, M, self.d_head)

        if mask is not None:
            mask = mx.expand_dims(mask, 1)
            mask = mx.broadcast_to(mask, (B, self.num_heads, N, M))
            mask = mask.reshape(B * self.num_heads, N, M)

        # Flash attention
        output, _ = flash_attention_forward(
            q, k, v, mask=mask, scale=1.0 / q.shape[-1] ** 0.5, causal=causal
        )

        # Reshape and project back
        output = output.reshape(B, self.num_heads, N, self.d_head).transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.out_proj(output)
