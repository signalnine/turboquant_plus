"""PolarQuant: Random rotation + optimal scalar quantization.

Algorithm 1 from the TurboQuant paper (AISTATS 2026).

After random rotation, coordinates follow a known Beta distribution (Gaussian in
high d), enabling optimal scalar quantization per coordinate independently.

Important: codebook is calibrated for unit-norm vectors. For non-unit-norm inputs,
we extract norms, normalize, quantize, then rescale on dequantization.
(Paper page 5: "store the L2 norms in floating-point precision and rescale")
"""

import numpy as np

from turboquant.codebook import optimal_centroids, nearest_centroid_indices
from turboquant.rotation import random_rotation_dense


class PolarQuant:
    """MSE-optimized vector quantizer via random rotation + scalar quantization.

    Handles arbitrary-norm vectors by extracting norms before quantization
    and rescaling after dequantization. This is critical for real KV cache
    tensors which are NOT unit-norm.

    Usage:
        pq = PolarQuant(d=128, bit_width=2, seed=42)
        indices, norms = pq.quantize(x)       # x: (d,) or (batch, d)
        x_hat = pq.dequantize(indices, norms)  # reconstructed

    Rotation methods:
        'dense'    — Haar QR rotation, O(d²), exact (default)
        'wht'      — Walsh-Hadamard + random signs, O(d log d), power-of-2 only
        'vilenkin' — Vilenkin-Hartley + random signs, O(d log d), any dimension
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42, rotation_method: str = 'dense'):
        self.d = d
        self.bit_width = bit_width
        self.n_centroids = 1 << bit_width
        self.rotation_method = rotation_method

        rng = np.random.default_rng(seed)

        if rotation_method == 'dense':
            self.rotation = random_rotation_dense(d, rng)
        elif rotation_method == 'wht':
            from turboquant.rotation import random_rotation_fast
            self.signs1, self.signs2, self.padded_d = random_rotation_fast(d, rng)
            self.rotation = None
        elif rotation_method == 'vilenkin':
            from turboquant.vilenkin import random_vilenkin_rotation
            self.signs1, self.signs2 = random_vilenkin_rotation(d, rng)
            self.rotation = None
        else:
            raise ValueError(f"Unknown rotation_method: {rotation_method!r}. "
                             f"Use 'dense', 'wht', or 'vilenkin'.")

        self.centroids = optimal_centroids(bit_width, d)

    def _rotate_forward(self, X: np.ndarray) -> np.ndarray:
        """Apply rotation to batch of vectors. Shape: (batch, d) → (batch, d)."""
        if self.rotation_method == 'dense':
            return (self.rotation @ X.T).T
        elif self.rotation_method == 'wht':
            from turboquant.rotation import apply_fast_rotation_batch
            return apply_fast_rotation_batch(X, self.signs1, self.signs2, self.padded_d)
        elif self.rotation_method == 'vilenkin':
            from turboquant.vilenkin import apply_vilenkin_rotation_batch
            return apply_vilenkin_rotation_batch(X, self.signs1, self.signs2)

    def _rotate_inverse(self, Y: np.ndarray) -> np.ndarray:
        """Apply inverse rotation. Shape: (batch, d) → (batch, d)."""
        if self.rotation_method == 'dense':
            return (self.rotation.T @ Y.T).T
        elif self.rotation_method == 'wht':
            # WHT is self-inverse: inverse = D1 @ H @ D2 (swap sign order)
            from turboquant.rotation import apply_fast_rotation_batch
            return apply_fast_rotation_batch(Y, self.signs2, self.signs1, self.padded_d)
        elif self.rotation_method == 'vilenkin':
            # VHT is self-inverse: inverse = D1 @ VHT @ D2 (swap sign order)
            from turboquant.vilenkin import apply_vilenkin_rotation_batch
            return apply_vilenkin_rotation_batch(Y, self.signs2, self.signs1)

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize a vector or batch of vectors.

        Args:
            x: Input vector(s), shape (d,) or (batch, d).

        Returns:
            (indices, norms) where:
                indices: integer indices, shape (d,) or (batch, d)
                norms: L2 norms, scalar or (batch,) — needed for dequantization
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        # Extract norms and normalize (paper page 5)
        norms = np.linalg.norm(x, axis=1)  # (batch,)
        # Avoid division by zero for zero vectors
        safe_norms = np.where(norms > 0, norms, 1.0)
        x_normalized = x / safe_norms[:, np.newaxis]

        # Rotate normalized vectors
        y = self._rotate_forward(x_normalized)

        # Nearest centroid per coordinate
        indices = nearest_centroid_indices(y, self.centroids)

        if single:
            return indices[0], norms[0]
        return indices, norms

    def dequantize(self, indices: np.ndarray, norms: np.ndarray) -> np.ndarray:
        """Dequantize indices back to vectors.

        Args:
            indices: Integer indices, shape (d,) or (batch, d).
            norms: Original L2 norms, scalar or (batch,).

        Returns:
            Reconstructed vectors, same shape as original input.
        """
        single = indices.ndim == 1
        if single:
            indices = indices[np.newaxis, :]
            norms = np.array([norms])

        # Look up centroids → unit-norm reconstruction
        y_hat = self.centroids[indices]
        x_hat_unit = self._rotate_inverse(y_hat)

        # Rescale by original norms
        x_hat = x_hat_unit * norms[:, np.newaxis]

        return x_hat[0] if single else x_hat

    def quantize_and_residual(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize and return indices, norms, and residual error.

        Used by TurboQuant's second stage (QJL on residual).

        Returns:
            (indices, norms, residual) where residual = x - dequantize(indices, norms).
        """
        indices, norms = self.quantize(x)
        x_hat = self.dequantize(indices, norms)
        residual = x - x_hat
        return indices, norms, residual
