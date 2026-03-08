"""
Negacyclic Number Theoretic Transform (NTT) implementation.

The negacyclic NTT computes polynomial evaluation at odd powers of a primitive
root. Given coefficients x[0], x[1], ..., x[N-1], the output is:

    y[k] = Σ_{n=0}^{N-1} x[n] · ψ^{(2k+1)·n}  (mod q)

where ψ is a primitive 2N-th root of unity (ψ^N ≡ -1 mod q).

This is equivalent to a cyclic NTT on "twisted" input, where each coefficient
x[n] is first multiplied by ψ^n.
"""

import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Modular Arithmetic
# -----------------------------------------------------------------------------
# a,b: jnp.ndarray
# q: int
# because x's value in [0, q), so we just need a conditional add/sub
def mod_add(a, b, q):
    """Return (a + b) mod q, elementwise."""
    q_u32 = jnp.uint32(q)
    res = a + b
    result = jnp.where(res >= q_u32, res - q_u32, res)
    return result.astype(jnp.uint32)


def mod_sub(a, b, q):
    """Return (a - b) mod q, elementwise."""
    # For uint32, need to handle underflow correctly
    # If a >= b, result is a - b
    # If a < b, result is (a + q) - b = q - (b - a)
    q_u32 = jnp.uint32(q)
    res = jnp.where(a >= b, a - b, q_u32 - (b - a))
    return res.astype(jnp.uint32)


def mod_mul(a, b, q):
    """Return (a * b) mod q, elementwise."""
    return (a.astype(jnp.int64) * b.astype(jnp.int64) % q).astype(jnp.uint32)

# -----------------------------------------------------------------------------
# Core NTT
# -----------------------------------------------------------------------------


def ntt(x, *, q, psi_powers, twiddles):
    """
    Compute the forward negacyclic NTT.

    Args:
        x: Input coefficients, shape (batch, N), values in [0, q)
        q: Prime modulus satisfying (q - 1) % 2N == 0
        psi_powers: Precomputed ψ^n table
        twiddles: Precomputed twiddle table

    Returns:
        jnp.ndarray: NTT output, same shape as input
    """
    batch, N = x.shape
    logN = N.bit_length() - 1
    
    # Step 1: Apply negacyclic twist: x'[n] = x[n] * psi^n (mod q)
    y = mod_mul(x, psi_powers, q)
    
    # Step 2: Bit-reversal permutation for DIT
    def bit_reverse_indices(n, logn):
        indices = jnp.arange(n)
        rev = jnp.zeros(n, dtype=jnp.int32)
        for b in range(logn):
            rev |= ((indices >> b) & 1) << (logn - 1 - b)
        return rev
    
    rev_indices = bit_reverse_indices(N, logN)
    y = y[:, rev_indices]
    
    # Step 3: Cooley-Tukey DIT butterfly stages
    # Process from small spans to large spans
    for stage in range(logN):
        span = 1 << stage  # Butterfly span: 1, 2, 4, ..., N/2
        
        # Get twiddle factors for this stage
        stage_twiddles = twiddles[span:2*span]
        
        # Reshape for vectorized operations
        # Each block has size 2*span and contains span butterfly pairs
        num_blocks = N // (2 * span)
        
        # Reshape into (batch, num_blocks, 2, span)
        # where dimension 2 separates the two halves of each block
        y_reshaped = y.reshape(batch, num_blocks, 2, span)
        
        # u and v are the two butterfly inputs
        u = y_reshaped[:, :, 0, :]  # Shape: (batch, num_blocks, span)
        v = y_reshaped[:, :, 1, :]  # Shape: (batch, num_blocks, span)
        
        # Broadcast twiddles to (1, 1, span) for broadcasting
        tw = stage_twiddles.reshape(1, 1, span)
        
        # DIT butterfly:
        # temp = twiddle * v
        # output_first = u + temp
        # output_second = u - temp
        temp = mod_mul(v, tw, q)
        y_reshaped = y_reshaped.at[:, :, 0, :].set(mod_add(u, temp, q))
        y_reshaped = y_reshaped.at[:, :, 1, :].set(mod_sub(u, temp, q))
        
        # Reshape back
        y = y_reshaped.reshape(batch, N)
    
    # Ensure final result is in [0, q) by doing one final reduction
    # This shouldn't be necessary if mod_add/mod_sub work correctly,
    # but let's be safe
    y = y % jnp.uint32(q)
    
    return y


def prepare_tables(*, q, psi_powers, twiddles):
    """
    Optional one-time table preparation.

    Override this if you want faster modular multiplication than JAX's "%".
    For example, you can convert the provided tables into Montgomery form
    (or any other domain) once here, then run `ntt` using your mod_mul.
    This function runs before timing, so its cost is not counted as latency.
    Must return (psi_powers, twiddles) in the form expected by `ntt`.
    """
    return psi_powers, twiddles

