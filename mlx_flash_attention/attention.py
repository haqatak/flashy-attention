import mlx.core as mx
import numpy as np
from typing import Tuple

def flash_attention_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array = None,
    scale: float = None,
    causal: bool = False,
    q_lens: mx.array = None,
    k_lens: mx.array = None,
    window_size: Tuple[int, int] = (-1, -1),
) -> tuple[mx.array, mx.array]:
    """
    An implementation of flash attention.

    Args:
        q (mx.array): The query vectors, in B x N x D.
        k (mx.array): The key vectors, in B x M x D.
        v (mx.array): The value vectors, in B x M x D_v.
        mask (mx.array, optional): The attention mask. It should be an array that
            can be broadcast to `(B, N, M)`. A value of 0 means attend, and a
            large negative number (e.g., -1e9) means mask. Defaults to None.
        scale (float, optional): The scale factor for the attention scores.
            Defaults to 1 / sqrt(D).
        causal (bool, optional): If True, apply a causal mask. Defaults to False.
        q_lens (mx.array, optional): The sequence lengths of the queries, in B.
            Defaults to None.
        k_lens (mx.array, optional): The sequence lengths of the keys, in B.
            Defaults to None.
        window_size (Tuple[int, int], optional): The size of the sliding window.
            Defaults to (-1, -1), which means no sliding window.

    Returns:
        (mx.array, mx.array): The output vectors and the log-sum-exp of the
        attention scores.
    """
    B, N, D = q.shape
    _, M, D_v = v.shape

    if causal:
        if N != M:
            raise ValueError("Causal masking requires query and key sequence lengths to be equal.")
        causal_mask = np.triu(np.full((N, M), -1e9), 1)
        causal_mask = mx.array(causal_mask, dtype=q.dtype)
        if mask is not None:
            mask = mask + causal_mask
        else:
            mask = causal_mask

    if q_lens is not None and k_lens is not None:
        q_mask = mx.arange(N)[None, :] < q_lens[:, None]
        k_mask = mx.arange(M)[None, :] < k_lens[:, None]
        varlen_mask = q_mask[:, :, None] * k_mask[:, None, :]
        varlen_mask = mx.where(varlen_mask, 0.0, float("-inf"))
        if mask is not None:
            mask = mask + varlen_mask
        else:
            mask = varlen_mask

    if window_size[0] != -1 or window_size[1] != -1:
        query_indices = mx.arange(N).reshape(-1, 1)
        key_indices = mx.arange(M).reshape(1, -1)

        if window_size[0] == -1:
            lower_bound_mask = mx.full((N, M), True)
        else:
            lower_bound_mask = key_indices >= (query_indices - window_size[0])

        if window_size[1] == -1:
            upper_bound_mask = mx.full((N, M), True)
        else:
            upper_bound_mask = key_indices <= (query_indices + window_size[1])

        sliding_window_mask = lower_bound_mask & upper_bound_mask
        sliding_window_mask = mx.where(sliding_window_mask, 0.0, float("-inf"))

        if mask is not None:
            mask = mask + sliding_window_mask
        else:
            mask = sliding_window_mask

    scale = scale or 1.0 / mx.sqrt(D)

    # Online softmax statistics
    o = mx.zeros((B, N, D_v), dtype=q.dtype)
    l = mx.zeros((B, N), dtype=q.dtype)
    m = mx.full((B, N), -mx.inf, dtype=q.dtype)

    # Tiling parameters
    # These would be tuned for performance on specific hardware.
    B_c = min(128, M)
    B_r = min(128, N)

    # Outer loop over keys and values
    for j in range(0, M, B_c):
        j_end = min(j + B_c, M)
        k_j = k[:, j:j_end, :]
        v_j = v[:, j:j_end, :]

        # Inner loop over queries
        for i in range(0, N, B_r):
            i_end = min(i + B_r, N)
            q_i = q[:, i:i_end, :]
            m_i = m[:, i:i_end]
            l_i = l[:, i:i_end]

            # Compute attention scores
            s_ij = (q_i @ k_j.transpose(0, 2, 1)) * scale

            if mask is not None:
                s_ij = s_ij + mask[:, i:i_end, j:j_end]

            # Online softmax update
            m_i_new = mx.maximum(m_i, mx.max(s_ij, axis=-1))
            p_ij = mx.exp(s_ij - m_i_new[:, :, None])

            l_i_new = mx.exp(m_i - m_i_new) * l_i + mx.sum(p_ij, axis=-1)

            # Update output
            o_i = o[:, i:i_end, :]
            o[:, i:i_end] = l_i[:, :, None] / l_i_new[:, :, None] * mx.exp(
                m_i - m_i_new
            )[:, :, None] * o_i + (p_ij @ v_j)

            # Update stats
            m[:, i:i_end] = m_i_new
            l[:, i:i_end] = l_i_new

    L = m + mx.log(l)
    return o, L


def _flash_attention_varlen_single(q, k, v, scale, causal, window_size):
    N, D = q.shape
    M, D_v = v.shape

    mask = None
    if causal:
        mask = np.triu(np.full((N, M), -1e9), 1)
        mask = mx.array(mask, dtype=q.dtype)

    if window_size[0] != -1 or window_size[1] != -1:
        query_indices = mx.arange(N).reshape(-1, 1)
        key_indices = mx.arange(M).reshape(1, -1)

        if window_size[0] == -1:
            lower_bound_mask = mx.full((N, M), True)
        else:
            lower_bound_mask = key_indices >= (query_indices - window_size[0])

        if window_size[1] == -1:
            upper_bound_mask = mx.full((N, M), True)
        else:
            upper_bound_mask = key_indices <= (query_indices + window_size[1])

        sliding_window_mask = lower_bound_mask & upper_bound_mask
        sliding_window_mask = mx.where(sliding_window_mask, 0.0, float("-inf"))

        if mask is not None:
            mask = mask + sliding_window_mask
        else:
            mask = sliding_window_mask

    o = mx.zeros((N, D_v), dtype=q.dtype)
    l = mx.zeros((N,), dtype=q.dtype)
    m = mx.full((N,), -mx.inf, dtype=q.dtype)

    B_c = min(128, M)
    B_r = min(128, N)

    for j in range(0, M, B_c):
        j_end = min(j + B_c, M)
        k_j = k[j:j_end, :]
        v_j = v[j:j_end, :]

        for i in range(0, N, B_r):
            i_end = min(i + B_r, N)
            q_i = q[i:i_end, :]
            m_i = m[i:i_end]
            l_i = l[i:i_end]

            s_ij = (q_i @ k_j.T) * scale

            if mask is not None:
                s_ij = s_ij + mask[i:i_end, j:j_end]

            m_i_new = mx.maximum(m_i, mx.max(s_ij, axis=-1))
            p_ij = mx.exp(s_ij - m_i_new[:, None])

            l_i_new = mx.exp(m_i - m_i_new) * l_i + mx.sum(p_ij, axis=-1)

            o_i = o[i:i_end, :]
            o[i:i_end] = (l_i[:, None] / l_i_new[:, None]) * mx.exp(m_i - m_i_new)[:, None] * o_i + (p_ij @ v_j)

            m[i:i_end] = m_i_new
            l[i:i_end] = l_i_new

    L = m + mx.log(l)
    return o, L

def flash_attention_varlen_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    cu_seqlens_q: mx.array,
    cu_seqlens_k: mx.array,
    scale: float,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
):
    B = len(cu_seqlens_q) - 1
    o = mx.zeros_like(q)

    for i in range(B):
        q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i+1]
        k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i+1]

        q_i = q[q_start:q_end]
        k_i = k[k_start:k_end]
        v_i = v[k_start:k_end]

        o_i, _ = _flash_attention_varlen_single(q_i, k_i, v_i, scale, causal, window_size)
        o[q_start:q_end] = o_i

    return o

def flash_attention_backward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    o: mx.array,
    L: mx.array,
    dO: mx.array,
    scale: float = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """
    An implementation of the backward pass of flash attention.

    Args:
        q (mx.array): The query vectors from the forward pass.
        k (mx.array): The key vectors from the forward pass.
        v (mx.array): The value vectors from the forward pass.
        o (mx.array): The output vectors from the forward pass.
        L (mx.array): The log-sum-exp of the attention scores from the forward pass.
        dO (mx.array): The gradient of the loss with respect to the output.
        scale (float, optional): The scale factor for the attention scores.
            Defaults to 1 / sqrt(D).

    Returns:
        (mx.array, mx.array, mx.array): The gradients with respect to Q, K, and V.
    """
    B, N, D = q.shape
    _, M, D_v = v.shape
    scale = scale or 1.0 / mx.sqrt(D)

    # Initialize gradients
    dQ = mx.zeros_like(q)
    dK = mx.zeros_like(k)
    dV = mx.zeros_like(v)

    # Tiling parameters
    B_c = min(128, M)
    B_r = min(128, N)

    # Backward pass for dQ
    D_stat = mx.sum(dO * o, axis=-1)
    for j in range(0, M, B_c):
        j_end = min(j + B_c, M)
        k_j = k[:, j:j_end, :]
        v_j = v[:, j:j_end, :]

        for i in range(0, N, B_r):
            i_end = min(i + B_r, N)
            q_i = q[:, i:i_end, :]
            L_i = L[:, i:i_end]
            dO_i = dO[:, i:i_end, :]
            D_i = D_stat[:, i:i_end]

            s_ij = (q_i @ k_j.transpose(0, 2, 1)) * scale
            p_ij = mx.exp(s_ij - L_i[:, :, None])
            dP_ij = dO_i @ v_j.transpose(0, 2, 1)
            dS_ij = p_ij * (dP_ij - D_i[:, :, None]) * scale
            dQ[:, i:i_end, :] += dS_ij @ k_j

    # Backward pass for dK and dV
    for i in range(0, N, B_r):
        i_end = min(i + B_r, N)
        q_i = q[:, i:i_end, :]
        L_i = L[:, i:i_end]
        dO_i = dO[:, i:i_end, :]
        D_i = D_stat[:, i:i_end]

        for j in range(0, M, B_c):
            j_end = min(j + B_c, M)
            k_j = k[:, j:j_end, :]
            v_j = v[:, j:j_end, :]

            s_ij = (q_i @ k_j.transpose(0, 2, 1)) * scale
            p_ij = mx.exp(s_ij - L_i[:, :, None])
            dP_ij = dO_i @ v_j.transpose(0, 2, 1)
            dS_ij = p_ij * (dP_ij - D_i[:, :, None]) * scale

            dK[:, j:j_end, :] += dS_ij.transpose(0, 2, 1) @ q_i
            dV[:, j:j_end, :] += p_ij.transpose(0, 2, 1) @ dO_i

    return dQ, dK, dV
