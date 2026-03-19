"""
Negacyclic Number Theoretic Transform (NTT) implementation.
"""

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


def mod_mul(a, b, q):
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