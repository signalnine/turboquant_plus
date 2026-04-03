"""Vilenkin-Hartley Transform for arbitrary-dimension decorrelation.

Generalizes Walsh-Hadamard (base-2 only) to mixed-radix dimensions via
the Vilenkin system on Z_{p1} × Z_{p2} × ... × Z_{pk}.

For power-of-2 dimensions, this reduces exactly to WHT.
For non-power-of-2 (e.g., 80 = 2^4 × 5), it applies mixed-radix
butterflies without zero-padding.

The Hartley variant keeps everything real-valued (cos+sin instead of
complex exponentials), preserving the self-inverse property needed for
TurboQuant's rotation scheme.

References:
    Vilenkin, N. Ya. "On a class of complete orthonormal systems." 1947.
    Fine, N. J. "On the Walsh functions." Trans. AMS, 1949.
"""

import numpy as np
from functools import lru_cache


def prime_factors(n: int) -> list[int]:
    """Return prime factorization as a list of primes (with repetition).

    Example: 80 → [2, 2, 2, 2, 5], 128 → [2, 2, 2, 2, 2, 2, 2]
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


@lru_cache(maxsize=32)
def _dft_matrix_real(p: int) -> np.ndarray:
    """Real-valued p×p DFT matrix (Hartley-style: cos + sin).

    For p=2, this is [[1, 1], [1, -1]] (standard WHT butterfly).
    For p>2, uses the Discrete Hartley Transform basis: cas(2πkn/p)
    where cas(θ) = cos(θ) + sin(θ).

    The DHT matrix is symmetric and involutory: H² = pI, so
    H/√p is orthogonal and self-inverse.
    """
    k = np.arange(p)
    n = np.arange(p)
    theta = 2.0 * np.pi * np.outer(k, n) / p
    # cas(θ) = cos(θ) + sin(θ), the Hartley kernel
    return np.cos(theta) + np.sin(theta)


def vilenkin_hartley_transform(x: np.ndarray) -> np.ndarray:
    """Vilenkin-Hartley Transform for arbitrary-length input.

    Applies mixed-radix butterfly stages based on prime factorization.
    For power-of-2 lengths, equivalent to Walsh-Hadamard Transform.

    Args:
        x: Input array of length n (n >= 1, any positive integer).

    Returns:
        Transformed array, normalized by 1/√n. Self-inverse.
    """
    n = len(x)
    if n < 1:
        raise ValueError(f"Input length must be >= 1, got {n}")
    if n == 1:
        return x.copy()

    x = x.copy().astype(np.float64)
    factors = prime_factors(n)

    # Mixed-radix in-place butterfly, one stage per prime factor.
    # Process factors from smallest to largest for numerical stability.
    stride = 1
    for p in factors:
        # At this stage, the array is viewed as groups of (stride * p) elements.
        # Within each group, apply p-point transform across stride-separated elements.
        if p == 2:
            # Optimized binary butterfly (same as WHT)
            for i in range(0, n, stride * 2):
                for j in range(stride):
                    a = x[i + j]
                    b = x[i + j + stride]
                    x[i + j] = a + b
                    x[i + j + stride] = a - b
        else:
            # General p-ary Hartley butterfly
            H = _dft_matrix_real(p)
            for i in range(0, n, stride * p):
                for j in range(stride):
                    # Gather p elements at stride-separated positions
                    indices = [i + j + k * stride for k in range(p)]
                    vals = np.array([x[idx] for idx in indices])
                    # Apply p-point Hartley transform
                    result = H @ vals
                    # Scatter back
                    for k in range(p):
                        x[indices[k]] = result[k]
        stride *= p

    return x / np.sqrt(n)


def vilenkin_hartley_transform_batch(X: np.ndarray) -> np.ndarray:
    """Batch Vilenkin-Hartley Transform. Shape: (batch, d).

    Vectorized for the common case where d is power-of-2 (pure WHT).
    Falls back to per-row for mixed-radix dimensions.
    """
    batch, d = X.shape
    factors = prime_factors(d)

    # If all factors are 2, use vectorized WHT (same as existing fast_walsh_hadamard)
    if all(f == 2 for f in factors):
        X = X.copy().astype(np.float64)
        h = 1
        while h < d:
            reshaped = X.reshape(batch, d // (h * 2), 2, h)
            a = reshaped[:, :, 0, :].copy()
            b = reshaped[:, :, 1, :].copy()
            reshaped[:, :, 0, :] = a + b
            reshaped[:, :, 1, :] = a - b
            X = reshaped.reshape(batch, d)
            h *= 2
        return X / np.sqrt(d)

    # Mixed-radix: per-row fallback
    result = np.zeros_like(X, dtype=np.float64)
    for i in range(batch):
        result[i] = vilenkin_hartley_transform(X[i])
    return result


def random_vilenkin_rotation(d: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate random sign vectors for Vilenkin-Hartley rotation.

    The rotation is: D1 @ VH @ D2 (random signs + Vilenkin-Hartley + random signs).
    Analogous to random_rotation_fast() but works for any dimension.

    Args:
        d: Dimension (any positive integer, not just power-of-2).
        rng: NumPy random generator.

    Returns:
        (signs1, signs2) — random ±1 vectors of length d.
    """
    signs1 = rng.choice([-1.0, 1.0], size=d)
    signs2 = rng.choice([-1.0, 1.0], size=d)
    return signs1, signs2


def apply_vilenkin_rotation(x: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Apply Vilenkin-Hartley random rotation to a vector.

    Rotation: x → D2 @ VHT @ D1 @ x
    No padding needed — works for any dimension.
    """
    d = len(x)
    y = x.copy().astype(np.float64)
    y *= signs1[:d]
    y = vilenkin_hartley_transform(y)
    y *= signs2[:d]
    return y


def apply_vilenkin_rotation_transpose(y: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Apply transpose (= inverse, since VHT is self-inverse and D is symmetric).

    Inverse: y → D1 @ VHT @ D2 @ y
    """
    d = len(y)
    x = y.copy().astype(np.float64)
    x *= signs2[:d]
    x = vilenkin_hartley_transform(x)
    x *= signs1[:d]
    return x


def apply_vilenkin_rotation_batch(X: np.ndarray, signs1: np.ndarray, signs2: np.ndarray) -> np.ndarray:
    """Apply Vilenkin-Hartley rotation to a batch. Shape: (batch, d)."""
    batch, d = X.shape
    Y = X.copy().astype(np.float64)
    Y *= signs1[np.newaxis, :d]
    Y = vilenkin_hartley_transform_batch(Y)
    Y *= signs2[np.newaxis, :d]
    return Y
