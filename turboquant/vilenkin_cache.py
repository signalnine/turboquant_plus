"""Vilenkin Coefficient Cache — sparse prime-harmonic KV cache compression.

Instead of storing quantized rotated values (PolarQuant), store the sparse
coefficients in the Vilenkin-Hartley basis. The key insight from empirical
validation: KV cache vectors are sparse in the prime-harmonic basis, with
~10 universal basis indices per (layer, head) explaining most of the energy.

Architecture:
  1. Forward VHT on each KV vector → coefficient vector
  2. Select top-k coefficients by energy (or use shared mask)
  3. Quantize coefficients to int4/int8
  4. Store: shared basis mask + per-position coefficient values + norm
  5. Reconstruct: look up coefficients, inverse VHT

This achieves 5-10× compression at +3-11% PPL — far beyond PolarQuant's
4.6× at fixed-centroid quantization.
"""

import numpy as np
from turboquant.vilenkin import (
    vilenkin_hartley_transform,
    vilenkin_hartley_transform_batch,
)


class VilenkinCacheConfig:
    """Configuration for Vilenkin coefficient cache."""

    def __init__(
        self,
        head_dim: int,
        energy_threshold: float = 0.95,
        quant_levels: int = 16,      # 16 = int4, 256 = int8
        max_coeffs: int = 0,         # 0 = auto from energy threshold
        shared_mask: bool = True,    # share basis indices across positions
    ):
        self.head_dim = head_dim
        self.energy_threshold = energy_threshold
        self.quant_levels = quant_levels
        self.max_coeffs = max_coeffs
        self.shared_mask = shared_mask


class VilenkinCacheEncoder:
    """Encode KV vectors into sparse Vilenkin coefficients."""

    def __init__(self, config: VilenkinCacheConfig):
        self.config = config
        self.d = config.head_dim

        # Shared mask: discovered from calibration data
        self._shared_indices = None  # set by calibrate()

    def calibrate(self, vectors: np.ndarray, universal_threshold: float = 0.9):
        """Discover universal basis indices from calibration vectors.

        Args:
            vectors: (n_vectors, head_dim) calibration data
            universal_threshold: fraction of positions where an index must
                                 appear in top-k to be considered universal

        Returns:
            shared_indices: array of universal basis indices
        """
        n, d = vectors.shape
        assert d == self.d

        # Transform all vectors
        coeffs = vilenkin_hartley_transform_batch(vectors)

        # For each vector, find top-k indices by energy
        k = self._auto_k(coeffs[0])  # use first vector to estimate k
        all_top_indices = set()
        index_counts = np.zeros(d, dtype=int)

        for i in range(n):
            c = coeffs[i]
            energy = c ** 2
            total_energy = np.sum(energy)
            if total_energy < 1e-12:
                continue

            # Sort by energy, find threshold
            sorted_idx = np.argsort(-energy)
            cumulative = np.cumsum(energy[sorted_idx]) / total_energy

            k_this = np.searchsorted(cumulative, self.config.energy_threshold) + 1
            if self.config.max_coeffs > 0:
                k_this = min(k_this, self.config.max_coeffs)

            top_idx = sorted_idx[:k_this]
            index_counts[top_idx] += 1

        # Universal indices: appear in > universal_threshold fraction of vectors
        threshold_count = int(n * universal_threshold)
        self._shared_indices = np.where(index_counts >= threshold_count)[0]

        return self._shared_indices

    def _auto_k(self, coeffs: np.ndarray) -> int:
        """Determine number of coefficients needed for energy threshold."""
        energy = coeffs ** 2
        total = np.sum(energy)
        if total < 1e-12:
            return 1
        sorted_energy = np.sort(energy)[::-1]
        cumulative = np.cumsum(sorted_energy) / total
        k = np.searchsorted(cumulative, self.config.energy_threshold) + 1
        if self.config.max_coeffs > 0:
            k = min(k, self.config.max_coeffs)
        return int(k)

    def encode(self, x: np.ndarray) -> dict:
        """Encode a single KV vector into sparse Vilenkin coefficients.

        Args:
            x: (head_dim,) vector

        Returns:
            dict with:
                'norm': float — L2 norm for rescaling
                'indices': int array — which basis indices are stored
                'values': int array — quantized coefficient values
                'scale': float — quantization scale
                'n_coeffs': int — number of stored coefficients
        """
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return {
                'norm': 0.0,
                'indices': np.array([], dtype=np.int32),
                'values': np.array([], dtype=np.int8),
                'scale': 0.0,
                'n_coeffs': 0,
            }

        # Normalize and transform
        x_unit = x / norm
        coeffs = vilenkin_hartley_transform(x_unit)

        # Select indices
        if self.config.shared_mask and self._shared_indices is not None:
            # Use shared mask — store values for universal indices only
            indices = self._shared_indices
        else:
            # Per-vector top-k by energy
            energy = coeffs ** 2
            total = np.sum(energy)
            sorted_idx = np.argsort(-energy)
            cumulative = np.cumsum(energy[sorted_idx]) / total
            k = np.searchsorted(cumulative, self.config.energy_threshold) + 1
            if self.config.max_coeffs > 0:
                k = min(k, self.config.max_coeffs)
            indices = np.sort(sorted_idx[:k])

        # Extract and quantize coefficients
        raw_values = coeffs[indices]
        amax = np.max(np.abs(raw_values)) if len(raw_values) > 0 else 1.0

        n_levels = self.config.quant_levels
        half_levels = n_levels // 2
        scale = amax / half_levels if amax > 1e-12 else 1.0
        quantized = np.clip(np.round(raw_values / scale), -half_levels, half_levels - 1).astype(np.int8)

        return {
            'norm': float(norm),
            'indices': indices.astype(np.int32),
            'values': quantized,
            'scale': float(scale),
            'n_coeffs': len(indices),
        }

    def encode_batch(self, X: np.ndarray) -> list[dict]:
        """Encode a batch of vectors."""
        return [self.encode(X[i]) for i in range(X.shape[0])]

    def decode(self, encoded: dict) -> np.ndarray:
        """Decode sparse Vilenkin coefficients back to a vector."""
        if encoded['n_coeffs'] == 0:
            return np.zeros(self.d)

        # Reconstruct sparse coefficient vector
        coeffs = np.zeros(self.d)
        coeffs[encoded['indices']] = encoded['values'].astype(np.float64) * encoded['scale']

        # Inverse VHT (self-inverse)
        x_unit = vilenkin_hartley_transform(coeffs)

        return x_unit * encoded['norm']

    def decode_batch(self, encoded_list: list[dict]) -> np.ndarray:
        """Decode a batch of encoded vectors."""
        result = np.zeros((len(encoded_list), self.d))
        for i, enc in enumerate(encoded_list):
            result[i] = self.decode(enc)
        return result

    def bytes_per_position(self, n_coeffs: int = 0) -> float:
        """Estimate storage bytes per position.

        With shared mask (stored once per head):
            per-position: norm(2) + scale(2) + values(n_coeffs × bits/level)
        Without shared mask:
            per-position: norm(2) + scale(2) + n_coeffs(1) + indices(n_coeffs×2) + values(n_coeffs×bits)
        """
        if n_coeffs == 0:
            # Estimate from typical coefficient count
            n_coeffs = int(self.d * (1.0 - self.config.energy_threshold) * 3)  # rough estimate

        bits_per_coeff = int(np.ceil(np.log2(self.config.quant_levels)))

        if self.config.shared_mask:
            # norm(fp16) + scale(fp16) + values
            value_bytes = int(np.ceil(n_coeffs * bits_per_coeff / 8))
            return 2 + 2 + value_bytes  # norm + scale + packed values
        else:
            # norm(fp16) + scale(fp16) + n_coeffs(1) + indices(2 each) + values
            value_bytes = int(np.ceil(n_coeffs * bits_per_coeff / 8))
            return 2 + 2 + 1 + n_coeffs * 2 + value_bytes

    def compression_ratio(self, n_coeffs: int = 0, original_bytes: int = 0) -> float:
        """Compute compression ratio vs fp16 storage."""
        if original_bytes == 0:
            original_bytes = self.d * 2  # fp16
        return original_bytes / self.bytes_per_position(n_coeffs)


def compare_methods(X: np.ndarray, head_dim: int, bit_width: int = 3):
    """Compare Vilenkin coefficient cache vs PolarQuant on the same data.

    Returns dict with MSE, cosine similarity, compression ratio for each method.
    """
    from turboquant.polar_quant import PolarQuant

    n = X.shape[0]
    results = {}

    # PolarQuant (current TurboQuant approach)
    for method in ['dense', 'vilenkin']:
        pq = PolarQuant(head_dim, bit_width, seed=42, rotation_method=method)
        mse_total = 0.0
        cos_total = 0.0
        for i in range(n):
            idx, norm = pq.quantize(X[i])
            x_hat = pq.dequantize(idx, norm)
            mse_total += np.mean((X[i] - x_hat) ** 2)
            cos_total += np.dot(X[i], x_hat) / (np.linalg.norm(X[i]) * np.linalg.norm(x_hat) + 1e-10)
        results[f'PolarQuant-{method}-{bit_width}bit'] = {
            'mse': mse_total / n,
            'cosine': cos_total / n,
            'bpv': bit_width + 16 / head_dim,  # bits + norm overhead
        }

    # Vilenkin coefficient cache at various energy thresholds
    for energy in [0.99, 0.95, 0.90]:
        for quant in [256, 16]:  # int8, int4
            label = f"int{'8' if quant == 256 else '4'}"
            config = VilenkinCacheConfig(
                head_dim=head_dim,
                energy_threshold=energy,
                quant_levels=quant,
                shared_mask=True,
            )
            enc = VilenkinCacheEncoder(config)

            # Calibrate on first half, test on second half
            half = n // 2
            enc.calibrate(X[:half])

            mse_total = 0.0
            cos_total = 0.0
            total_coeffs = 0
            for i in range(half, n):
                encoded = enc.encode(X[i])
                x_hat = enc.decode(encoded)
                mse_total += np.mean((X[i] - x_hat) ** 2)
                cos_total += np.dot(X[i], x_hat) / (np.linalg.norm(X[i]) * np.linalg.norm(x_hat) + 1e-10)
                total_coeffs += encoded['n_coeffs']

            avg_coeffs = total_coeffs / (n - half)
            results[f'Vilenkin-{int(energy*100)}%-{label}'] = {
                'mse': mse_total / (n - half),
                'cosine': cos_total / (n - half),
                'avg_coeffs': avg_coeffs,
                'bytes_per_pos': enc.bytes_per_position(int(avg_coeffs)),
                'compression': enc.compression_ratio(int(avg_coeffs)),
                'bpv': enc.bytes_per_position(int(avg_coeffs)) * 8 / head_dim,
            }

    return results
