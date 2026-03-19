"""
Negacyclic Number Theoretic Transform (NTT) implementation.
"""
# AGENT ATTENTION!
# ADITIONAL JIT IS NOT ALLOWED!

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------

def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    res = a + b
    return jnp.where(res >= q, res - q, res)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    q = jnp.asarray(q, dtype=jnp.uint32)
    return jnp.where(a >= b, a - b, a + q - b)


def mod_mul(a: jnp.ndarray, b: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Return (a * b) mod q, elementwise."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    prod = a.astype(jnp.uint64) * b.astype(jnp.uint64)
    return jnp.remainder(prod, q64).astype(jnp.uint32)


@lru_cache(maxsize=None)
def _bit_reverse_indices(N):
    """Return bit-reversal permutation indices for power-of-two N."""
    logN = N.bit_length() - 1
    idx = np.arange(N, dtype=np.int32)
    rev = np.zeros(N, dtype=np.int32)
    for b in range(logN):
        rev |= ((idx >> b) & 1) << (logN - 1 - b)
    return jnp.asarray(rev, dtype=jnp.int32)


def _unpack_twiddles(twiddles):
    """Support both raw twiddle arrays and prepare_tables packed format."""
    if isinstance(twiddles, dict):
        return (
            twiddles.get("rev"),
            twiddles.get("psi_rev"),
            twiddles.get("stage_twiddles"),
            twiddles.get("raw"),
        )
    return None, None, None, twiddles


# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------


@jax.jit
def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed psi^n table
        twiddles: Precomputed twiddle table or packed prepare_tables format

    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    return ntt_candidate1(x, q=q, psi_powers=psi_powers, twiddles=twiddles)

def prepare_tables(*, q, psi_powers, twiddles):
    """Optional one-time table preparation."""
    del q

    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)

    N = int(psi_powers.shape[0])
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a positive power of two, got {N}")

    logN = N.bit_length() - 1
    rev = _bit_reverse_indices(N)

    # Precompute:
    # 1) psi_powers in bit-reversed order, so twist+permute can be fused
    # 2) stage twiddles already reshaped for broadcast
    psi_rev = psi_powers[rev]
    stage_twiddles = tuple(
        twiddles[1 << s: 1 << (s + 1)].reshape(1, 1, 1 << s)
        for s in range(logN)
    )

    packed = {
        "rev": rev,
        "psi_rev": psi_rev,
        "stage_twiddles": stage_twiddles,
    }
    return psi_powers, packed

# 4.54s, 4.84s, 4.91s, 4.44s
def ntt_candidate0(x, *, q, psi_powers, twiddles):
    x = jnp.asarray(x, dtype=jnp.uint32)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)

    batch, N = x.shape
    logN = N.bit_length() - 1

    rev, psi_rev, stage_twiddles, raw_twiddles = _unpack_twiddles(twiddles)

    # Fast path:
    # fuse the negacyclic twist with the bit-reversal gather
    # so we avoid:
    #   y = mod_mul(x, psi_powers, q)
    #   y = y[:, rev]
    if rev is not None and psi_rev is not None:
        y = mod_mul(x[:, rev], psi_rev, q)
    else:
        if rev is None:
            rev = _bit_reverse_indices(N)
        y = mod_mul(x, psi_powers, q)
        y = y[:, rev]

    # Butterfly stages
    for stage in range(logN):
        span = 1 << stage
        block = 2 * span
        num_blocks = N // block

        if stage_twiddles is None:
            tw = jnp.asarray(raw_twiddles[span:block], dtype=jnp.uint32).reshape(1, 1, span)
        else:
            tw = stage_twiddles[stage]

        y_reshaped = y.reshape(batch, num_blocks, 2, span)
        u = y_reshaped[:, :, 0, :]
        v = y_reshaped[:, :, 1, :]

        t = mod_mul(v, tw, q)
        top = mod_add(u, t, q)
        bot = mod_sub(u, t, q)

        y = jnp.stack((top, bot), axis=2).reshape(batch, N)

    return y

# 4.34s, 4.41s, 4.39s, 4.26s
def ntt_candidate1(x, *, q, psi_powers, twiddles):
    x = jnp.asarray(x, dtype=jnp.uint32)
    q = jnp.asarray(q, dtype=jnp.uint32)
    q64 = q.astype(jnp.uint64) # 提前转 u64 减少循环内转换

    batch, N = x.shape
    logN = N.bit_length() - 1

    rev, psi_rev, stage_twiddles, raw_twiddles = _unpack_twiddles(twiddles)

    # 1. Fused Pre-processing
    if rev is not None and psi_rev is not None:
        y = jnp.remainder(x[:, rev].astype(jnp.uint64) * psi_rev.astype(jnp.uint64), q64).astype(jnp.uint32)
    else:
        if rev is None:
            rev = _bit_reverse_indices(N)
        y = jnp.remainder(x.astype(jnp.uint64) * psi_powers.astype(jnp.uint64), q64).astype(jnp.uint32)
        y = y[:, rev]

    # 2. 优化后的蝴蝶运算循环
    for stage in range(logN):
        span = 1 << stage
        block = 2 * span
        num_blocks = N // block

        if stage_twiddles is None:
            tw = jnp.asarray(raw_twiddles[span:block], dtype=jnp.uint32).reshape(1, 1, span)
        else:
            tw = stage_twiddles[stage]

        # 改变视角：将 y 视为 (batch, num_blocks, 2, span)
        # 但我们不直接 stack，而是通过切片直接计算
        y_reshaped = y.reshape(batch, num_blocks, 2, span)
        
        u = y_reshaped[:, :, 0, :]
        v = y_reshaped[:, :, 1, :]

        # 这里的 mul + remainder 会被 XLA 自动转为高效的 Barrett/MagicMul
        t = jnp.remainder(v.astype(jnp.uint64) * tw.astype(jnp.uint64), q64).astype(jnp.uint32)

        top = jnp.where(u + t >= q, u + t - q, u + t)
        bot = jnp.where(u >= t, u - t, u + q - t)

        # 【关键优化点】：利用 concatenate 取代 stack
        # 在很多后端，concatenate(..., axis=2) 配合 reshape 比 stack 更有利于内存对齐
        y = jnp.concatenate([top[:, :, jnp.newaxis, :], bot[:, :, jnp.newaxis, :]], axis=2).reshape(batch, N)

    return y