import mlx.core as mx


def flash_attention_forward(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    mask: mx.array = None,
    scale: float = None,
    causal: bool = False,
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

    Returns:
        (mx.array, mx.array): The output vectors and the log-sum-exp of the
        attention scores.
    """
    B, N, D = q.shape
    _, M, D_v = v.shape

    if causal:
        if N != M:
            raise ValueError("Causal masking requires query and key sequence lengths to be equal.")

    scale = scale or 1.0 / mx.sqrt(D)

    # Online softmax statistics
    o = mx.zeros((B, N, D_v), dtype=q.dtype)
    l = mx.zeros((B, N))
    m = mx.full((B, N), -mx.inf)

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

            if causal:
                query_indices = mx.arange(i, i_end).reshape(-1, 1)
                key_indices = mx.arange(j, j_end).reshape(1, -1)
                causal_mask_block = key_indices > query_indices
                s_ij = mx.where(causal_mask_block, float("-inf"), s_ij)

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
