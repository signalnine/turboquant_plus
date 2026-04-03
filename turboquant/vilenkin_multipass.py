"""Multi-pass progressive Vilenkin coefficient cache.

Each pass uses a different random rotation (seeded by a different prime),
then selects top coefficients from the VHT of the rotated residual.
Different rotations capture different structural components:

  Pass 1 (seed=2): Captures the dominant structure (skeleton)
  Pass 2 (seed=3): Captures what pass 1 missed (detail)
  Pass 3 (seed=5): Captures remaining fine structure (texture)

Each subsequent pass operates on the RESIDUAL from previous passes,
with decreasing energy thresholds.

From Position_Is_Arithmetic research:
  Single-pass int8: 5.1× compression, cos ~0.95
  Multi-pass 3 passes: 6.23× compression at same quality
"""

import numpy as np
from turboquant.vilenkin import (
    vilenkin_hartley_transform,
    random_vilenkin_rotation,
    apply_vilenkin_rotation,
    apply_vilenkin_rotation_transpose,
)


class MultiPassConfig:
    """Configuration for multi-pass Vilenkin encoding."""

    def __init__(
        self,
        head_dim: int,
        passes: list[dict] | None = None,
        quant_levels: int = 16,
    ):
        self.head_dim = head_dim
        self.quant_levels = quant_levels

        if passes is None:
            # Default: 3 passes with decreasing energy thresholds
            # Seeds are primes to get maximally different rotations
            self.passes = [
                {'seed': 2,  'energy_threshold': 0.95, 'max_coeffs': 16},
                {'seed': 3,  'energy_threshold': 0.80, 'max_coeffs': 32},
                {'seed': 5,  'energy_threshold': 0.65, 'max_coeffs': 24},
            ]
        else:
            self.passes = passes


class MultiPassEncoder:
    """Multi-pass progressive Vilenkin encoder."""

    def __init__(self, config: MultiPassConfig):
        self.config = config
        self.d = config.head_dim

        # Pre-generate rotation sign arrays for each pass
        self.rotations = []
        for p in config.passes:
            rng = np.random.default_rng(p['seed'])
            s1, s2 = random_vilenkin_rotation(self.d, rng)
            self.rotations.append((s1, s2))

    def encode(self, x: np.ndarray) -> dict:
        """Multi-pass encode a single vector.

        Returns dict with per-pass coefficient data + shared metadata.
        """
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return {'norm': 0.0, 'passes': [], 'n_total_coeffs': 0}

        x_unit = x / norm
        residual = x_unit.copy()
        passes = []

        for i, (pcfg, (s1, s2)) in enumerate(zip(self.config.passes, self.rotations)):
            # Rotate residual with this pass's signs
            rotated = residual * s1
            coeffs = vilenkin_hartley_transform(rotated)
            coeffs = coeffs * s2

            # Select top-k by energy
            energy = coeffs ** 2
            total_energy = np.sum(energy)
            if total_energy < 1e-12:
                break

            sorted_idx = np.argsort(-energy)
            cumulative = np.cumsum(energy[sorted_idx]) / total_energy

            k = np.searchsorted(cumulative, pcfg['energy_threshold']) + 1
            k = min(k, pcfg.get('max_coeffs', k))
            k = min(k, self.d)

            top_idx = np.sort(sorted_idx[:k])
            raw_values = coeffs[top_idx]

            # Quantize
            amax = np.max(np.abs(raw_values)) if len(raw_values) > 0 else 1.0
            half_levels = self.config.quant_levels // 2
            scale = amax / half_levels if amax > 1e-12 else 1.0
            inv_scale = half_levels / amax if amax > 1e-12 else 0.0
            quantized = np.clip(np.round(raw_values * inv_scale), -half_levels, half_levels - 1).astype(np.int8)

            # Reconstruct this pass and subtract from residual
            recon_coeffs = np.zeros(self.d)
            recon_coeffs[top_idx] = quantized.astype(np.float64) * scale

            # Inverse: undo rotation
            recon_rotated = recon_coeffs * s2
            recon = vilenkin_hartley_transform(recon_rotated)
            recon = recon * s1

            residual = residual - recon

            passes.append({
                'indices': top_idx.astype(np.uint16),
                'values': quantized,
                'scale': float(scale),
                'n_coeffs': len(top_idx),
                'energy_captured': float(np.sum(energy[top_idx]) / total_energy),
            })

        total_coeffs = sum(p['n_coeffs'] for p in passes)
        return {
            'norm': float(norm),
            'passes': passes,
            'n_total_coeffs': total_coeffs,
            'residual_norm': float(np.linalg.norm(residual)),
        }

    def decode(self, encoded: dict) -> np.ndarray:
        """Multi-pass decode."""
        if encoded['n_total_coeffs'] == 0:
            return np.zeros(self.d)

        result = np.zeros(self.d)

        for i, (pass_data, (s1, s2)) in enumerate(zip(encoded['passes'], self.rotations)):
            # Reconstruct coefficients
            coeffs = np.zeros(self.d)
            coeffs[pass_data['indices']] = pass_data['values'].astype(np.float64) * pass_data['scale']

            # Inverse rotation
            recon = coeffs * s2
            recon = vilenkin_hartley_transform(recon)
            recon = recon * s1

            result += recon

        return result * encoded['norm']

    def bytes_per_position(self, encoded: dict) -> float:
        """Estimate storage bytes for one position."""
        total = 2 + 1  # norm(fp16) + n_passes(uint8)
        bits_per_coeff = int(np.ceil(np.log2(self.config.quant_levels)))

        for p in encoded['passes']:
            # Per pass: scale(fp16) + n_coeffs(uint8) + indices(uint16 each) + values(packed)
            total += 2 + 1  # scale + n_coeffs
            total += p['n_coeffs'] * 2  # indices
            total += int(np.ceil(p['n_coeffs'] * bits_per_coeff / 8))  # packed values

        return total

    def compression_ratio(self, encoded: dict) -> float:
        return (self.d * 2) / self.bytes_per_position(encoded)


def compare_single_vs_multi(X: np.ndarray, head_dim: int):
    """Compare single-pass vs multi-pass on real data."""

    n = X.shape[0]

    # Single-pass VHT (our current approach)
    from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder

    for n_coeffs in [48, 64, 96]:
        config = VilenkinCacheConfig(head_dim, energy_threshold=0.99, quant_levels=16,
                                      max_coeffs=n_coeffs, shared_mask=False)
        enc = VilenkinCacheEncoder(config)
        cos_total = 0
        bytes_total = 0
        for i in range(n):
            encoded = enc.encode(X[i])
            x_hat = enc.decode(encoded)
            cos_total += np.dot(X[i], x_hat) / (np.linalg.norm(X[i]) * np.linalg.norm(x_hat) + 1e-10)
            bytes_total += 4 + n_coeffs * 2 + n_coeffs // 2  # overhead + indices + int4 values
        avg_cos = cos_total / n
        avg_bytes = bytes_total / n
        ratio = (head_dim * 2) / avg_bytes
        print(f"Single-pass {n_coeffs:>3} coeffs int4: cos={avg_cos:.4f}  bytes={avg_bytes:.0f}  {ratio:.1f}x")

    # Multi-pass
    for max_total in [48, 64, 72]:
        # Distribute coefficients across 3 passes
        k1 = max(4, max_total // 5)
        k2 = max(8, max_total * 2 // 5)
        k3 = max_total - k1 - k2
        passes = [
            {'seed': 2, 'energy_threshold': 0.95, 'max_coeffs': k1},
            {'seed': 3, 'energy_threshold': 0.80, 'max_coeffs': k2},
            {'seed': 5, 'energy_threshold': 0.65, 'max_coeffs': k3},
        ]
        config = MultiPassConfig(head_dim, passes=passes, quant_levels=16)
        mp_enc = MultiPassEncoder(config)

        cos_total = 0
        bytes_total = 0
        coeffs_total = 0
        for i in range(n):
            encoded = mp_enc.encode(X[i])
            x_hat = mp_enc.decode(encoded)
            cos_total += np.dot(X[i], x_hat) / (np.linalg.norm(X[i]) * np.linalg.norm(x_hat) + 1e-10)
            bytes_total += mp_enc.bytes_per_position(encoded)
            coeffs_total += encoded['n_total_coeffs']

        avg_cos = cos_total / n
        avg_bytes = bytes_total / n
        avg_coeffs = coeffs_total / n
        ratio = (head_dim * 2) / avg_bytes
        print(f"Multi-pass  {max_total:>3} budget (k={k1}+{k2}+{k3}): cos={avg_cos:.4f}  coeffs={avg_coeffs:.0f}  bytes={avg_bytes:.0f}  {ratio:.1f}x")
