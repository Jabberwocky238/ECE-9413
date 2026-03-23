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
def montgomery_neg_inv32(q: int) -> int:
    """Return -q^{-1} mod 2^32 for an odd 32-bit modulus q."""
    if q <= 0 or q >= (1 << 32):
        raise ValueError(f"q must fit in 32 bits, got {q}")
    if (q & 1) == 0:
        raise ValueError(f"q must be odd for Montgomery arithmetic, got {q}")
    return (-pow(q, -1, 1 << 32)) & 0xFFFFFFFF


def montgomery_neg_inv32_jax(q: jnp.ndarray) -> jnp.ndarray:
    """Return -q^{-1} mod 2^32 using only JAX ops."""
    q32 = jnp.asarray(q, dtype=jnp.uint32)
    inv = jnp.asarray(1, dtype=jnp.uint32)

    # Newton iteration over 2-adics: each step doubles the number of
    # correct low bits of q^{-1} modulo 2^32.
    for _ in range(5):
        inv = inv * (jnp.asarray(2, dtype=jnp.uint32) - q32 * inv)

    return jnp.asarray(0, dtype=jnp.uint32) - inv


def montgomery_reduce(
    t: jnp.ndarray,
    *,
    q: jnp.ndarray,
    q_neg_inv: int | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return t * R^{-1} mod q using Montgomery reduction with R = 2^32."""
    q32 = jnp.asarray(q, dtype=jnp.uint32)
    q64 = q32.astype(jnp.uint64)

    if q_neg_inv is None:
        q_neg_inv = montgomery_neg_inv32_jax(q32)

    t64 = jnp.asarray(t, dtype=jnp.uint64)
    q_neg_inv64 = jnp.asarray(q_neg_inv, dtype=jnp.uint64)
    mask = jnp.asarray((1 << 32) - 1, dtype=jnp.uint64)

    m = (t64 * q_neg_inv64) & mask
    u = (t64 + m * q64) >> 32
    return jnp.where(u >= q64, u - q64, u).astype(jnp.uint32)


def montgomery_mul(
    a: jnp.ndarray,
    b: jnp.ndarray,
    *,
    q: jnp.ndarray,
    q_neg_inv: int | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return a * b * R^{-1} mod q in the Montgomery domain."""
    prod = jnp.asarray(a, dtype=jnp.uint64) * jnp.asarray(b, dtype=jnp.uint64)
    return montgomery_reduce(prod, q=q, q_neg_inv=q_neg_inv)


def to_montgomery(a: jnp.ndarray, *, q: jnp.ndarray) -> jnp.ndarray:
    """Return a * R mod q with R = 2^32."""
    q64 = jnp.asarray(q, dtype=jnp.uint64)
    a64 = jnp.asarray(a, dtype=jnp.uint64)
    r_mod_q = jnp.remainder(jnp.asarray(1 << 32, dtype=jnp.uint64), q64)
    return jnp.remainder(a64 * r_mod_q, q64).astype(jnp.uint32)


def from_montgomery(
    a: jnp.ndarray,
    *,
    q: jnp.ndarray,
    q_neg_inv: int | jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return the standard representation of a Montgomery-domain value."""
    return montgomery_reduce(a.astype(jnp.uint64), q=q, q_neg_inv=q_neg_inv)


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
    return prepare_tables0(q=q, psi_powers=psi_powers, twiddles=twiddles)

def prepare_tables0(*, q, psi_powers, twiddles):
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


def prepare_tables2(*, q, psi_powers, twiddles):
    """Prepare standard and Montgomery-domain tables for candidate2-style NTT."""
    q32 = jnp.asarray(q, dtype=jnp.uint32)
    psi_powers = jnp.asarray(psi_powers, dtype=jnp.uint32)
    twiddles = jnp.asarray(twiddles, dtype=jnp.uint32)

    N = int(psi_powers.shape[0])
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(f"N must be a positive power of two, got {N}")

    logN = N.bit_length() - 1
    rev = _bit_reverse_indices(N)

    psi_rev = psi_powers[rev]
    stage_twiddles = tuple(
        twiddles[1 << s: 1 << (s + 1)].reshape(1, 1, 1 << s)
        for s in range(logN)
    )

    q_neg_inv = montgomery_neg_inv32(int(q32))
    r_mod_q = np.uint32((1 << 32) % int(q32))
    psi_rev_mont = jnp.remainder(
        psi_rev.astype(jnp.uint64) * jnp.asarray(r_mod_q, dtype=jnp.uint64),
        q32.astype(jnp.uint64),
    ).astype(jnp.uint32)
    stage_twiddles_mont = tuple(
        jnp.remainder(
            tw.astype(jnp.uint64) * jnp.asarray(r_mod_q, dtype=jnp.uint64),
            q32.astype(jnp.uint64),
        ).astype(jnp.uint32)
        for tw in stage_twiddles
    )

    packed = {
        "rev": rev,
        "psi_rev": psi_rev,
        "psi_rev_mont": psi_rev_mont,
        "stage_twiddles": stage_twiddles,
        "stage_twiddles_mont": stage_twiddles_mont,
        "q_neg_inv": jnp.asarray(q_neg_inv, dtype=jnp.uint32),
        "r_mod_q": jnp.asarray(r_mod_q, dtype=jnp.uint32),
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


def ntt_candidate2(x, *, q, psi_powers, twiddles):
    """Forward negacyclic NTT using Montgomery multiplication in butterflies."""
    del psi_powers
    x = jnp.asarray(x, dtype=jnp.uint32)
    q = jnp.asarray(q, dtype=jnp.uint32)

    batch, N = x.shape
    logN = N.bit_length() - 1

    rev = twiddles["rev"]
    psi_rev_mont = twiddles["psi_rev_mont"]
    twiddle_source = twiddles["stage_twiddles_mont"]
    q_neg_inv = twiddles["q_neg_inv"]

    y = mod_mul(x[:, rev], psi_rev_mont, q)

    for stage in range(logN):
        span = 1 << stage
        block = 2 * span
        num_blocks = N // block
        tw_mont = jnp.asarray(twiddle_source[stage], dtype=jnp.uint32)

        y_reshaped = y.reshape(batch, num_blocks, 2, span)
        u = y_reshaped[:, :, 0, :]
        v = y_reshaped[:, :, 1, :]

        t = montgomery_mul(v, tw_mont, q=q, q_neg_inv=q_neg_inv)
        top = mod_add(u, t, q)
        bot = mod_sub(u, t, q)

        y = jnp.concatenate(
            [top[:, :, jnp.newaxis, :], bot[:, :, jnp.newaxis, :]],
            axis=2,
        ).reshape(batch, N)

    return from_montgomery(y, q=q, q_neg_inv=q_neg_inv)
