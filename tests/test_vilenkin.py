"""Tests for the Vilenkin-Hartley Transform."""

import numpy as np
import pytest


class TestPrimeFactors:
    def test_power_of_2(self):
        from turboquant.vilenkin import prime_factors
        assert prime_factors(128) == [2] * 7

    def test_mixed(self):
        from turboquant.vilenkin import prime_factors
        assert prime_factors(80) == [2, 2, 2, 2, 5]

    def test_prime(self):
        from turboquant.vilenkin import prime_factors
        assert prime_factors(7) == [7]

    def test_one(self):
        from turboquant.vilenkin import prime_factors
        assert prime_factors(1) == []


class TestVilenkinHartleyTransform:
    """Core transform properties."""

    def test_self_inverse(self):
        """VHT applied twice (with normalization) should return the original."""
        from turboquant.vilenkin import vilenkin_hartley_transform
        rng = np.random.default_rng(42)
        for d in [32, 64, 80, 128, 256]:
            x = rng.standard_normal(d)
            y = vilenkin_hartley_transform(x)
            x_back = vilenkin_hartley_transform(y)
            np.testing.assert_allclose(x_back, x, atol=1e-10,
                err_msg=f"Self-inverse failed for d={d}")

    def test_matches_wht_for_power_of_2(self):
        """For d=2^k, VHT should match Walsh-Hadamard."""
        from turboquant.vilenkin import vilenkin_hartley_transform
        from turboquant.rotation import fast_walsh_hadamard_transform
        rng = np.random.default_rng(42)
        for d in [32, 64, 128, 256]:
            x = rng.standard_normal(d)
            vht = vilenkin_hartley_transform(x)
            wht = fast_walsh_hadamard_transform(x)
            np.testing.assert_allclose(vht, wht, atol=1e-10,
                err_msg=f"VHT != WHT for d={d}")

    def test_orthogonality(self):
        """Transform should preserve L2 norm (Parseval's theorem)."""
        from turboquant.vilenkin import vilenkin_hartley_transform
        rng = np.random.default_rng(42)
        for d in [32, 64, 80, 128, 60, 96]:
            x = rng.standard_normal(d)
            y = vilenkin_hartley_transform(x)
            np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x),
                rtol=1e-10, err_msg=f"Norm not preserved for d={d}")

    def test_non_power_of_2_dimensions(self):
        """VHT should work for dimensions that WHT cannot handle."""
        from turboquant.vilenkin import vilenkin_hartley_transform
        rng = np.random.default_rng(42)
        for d in [80, 60, 96, 48, 72, 120, 192]:
            x = rng.standard_normal(d)
            y = vilenkin_hartley_transform(x)
            # Self-inverse
            x_back = vilenkin_hartley_transform(y)
            np.testing.assert_allclose(x_back, x, atol=1e-10,
                err_msg=f"Self-inverse failed for d={d}")
            # Norm preservation
            np.testing.assert_allclose(np.linalg.norm(y), np.linalg.norm(x),
                rtol=1e-10, err_msg=f"Norm not preserved for d={d}")

    def test_head_dim_80(self):
        """Specific test for head_dim=80 (Qwen3-4B), the motivating case."""
        from turboquant.vilenkin import vilenkin_hartley_transform
        rng = np.random.default_rng(42)
        x = rng.standard_normal(80)
        y = vilenkin_hartley_transform(x)
        x_back = vilenkin_hartley_transform(y)
        np.testing.assert_allclose(x_back, x, atol=1e-10)


class TestVilenkinRotation:
    """Random rotation (signs + VHT + signs) properties."""

    def test_rotation_is_orthogonal(self):
        """Rotation should preserve inner products."""
        from turboquant.vilenkin import random_vilenkin_rotation, apply_vilenkin_rotation
        rng = np.random.default_rng(42)
        for d in [64, 80, 128]:
            signs1, signs2 = random_vilenkin_rotation(d, rng)
            x = rng.standard_normal(d)
            z = rng.standard_normal(d)
            rx = apply_vilenkin_rotation(x, signs1, signs2)
            rz = apply_vilenkin_rotation(z, signs1, signs2)
            np.testing.assert_allclose(np.dot(rx, rz), np.dot(x, z),
                rtol=1e-10, err_msg=f"Inner product not preserved for d={d}")

    def test_rotation_inverse(self):
        """Applying rotation then transpose should recover original."""
        from turboquant.vilenkin import (random_vilenkin_rotation,
            apply_vilenkin_rotation, apply_vilenkin_rotation_transpose)
        rng = np.random.default_rng(42)
        for d in [64, 80, 128]:
            signs1, signs2 = random_vilenkin_rotation(d, rng)
            x = rng.standard_normal(d)
            rx = apply_vilenkin_rotation(x, signs1, signs2)
            x_back = apply_vilenkin_rotation_transpose(rx, signs1, signs2)
            np.testing.assert_allclose(x_back, x, atol=1e-10,
                err_msg=f"Rotation inverse failed for d={d}")

    def test_matches_wht_rotation_for_power_of_2(self):
        """For d=2^k, Vilenkin rotation should match WHT rotation."""
        from turboquant.vilenkin import random_vilenkin_rotation, apply_vilenkin_rotation
        from turboquant.rotation import apply_fast_rotation
        rng_v = np.random.default_rng(42)
        rng_w = np.random.default_rng(42)
        d = 128
        signs1_v, signs2_v = random_vilenkin_rotation(d, rng_v)
        # WHT rotation uses same rng seed, so signs match
        signs1_w, signs2_w, padded_d = rng_w.choice([-1.0, 1.0], size=d), rng_w.choice([-1.0, 1.0], size=d), d

        x = np.random.default_rng(99).standard_normal(d)
        rv = apply_vilenkin_rotation(x, signs1_v, signs2_v)
        # WHT rotation with same signs
        rw = apply_fast_rotation(x, signs1_v, signs2_v, d)
        np.testing.assert_allclose(rv, rw, atol=1e-10)


class TestVilenkinBatch:
    """Batch transform tests."""

    def test_batch_matches_single(self):
        from turboquant.vilenkin import vilenkin_hartley_transform, vilenkin_hartley_transform_batch
        rng = np.random.default_rng(42)
        for d in [64, 80, 128]:
            X = rng.standard_normal((16, d))
            Y_batch = vilenkin_hartley_transform_batch(X)
            for i in range(16):
                y_single = vilenkin_hartley_transform(X[i])
                np.testing.assert_allclose(Y_batch[i], y_single, atol=1e-10,
                    err_msg=f"Batch mismatch at row {i}, d={d}")


class TestVilenkinQuantizationQuality:
    """Compare VHT vs WHT decorrelation quality for quantization."""

    @pytest.mark.parametrize("bit_width", [2, 3, 4])
    def test_mse_with_vilenkin_rotation(self, bit_width):
        """VHT rotation + Lloyd-Max quantization should achieve reasonable MSE."""
        from turboquant.vilenkin import random_vilenkin_rotation, apply_vilenkin_rotation, apply_vilenkin_rotation_transpose
        from turboquant.codebook import optimal_centroids, nearest_centroid_indices

        d = 128
        rng = np.random.default_rng(42)
        signs1, signs2 = random_vilenkin_rotation(d, rng)
        centroids = optimal_centroids(bit_width, d)

        # Generate test vectors (Gaussian, like KV cache entries)
        n_vectors = 100
        X = rng.standard_normal((n_vectors, d))

        total_mse = 0.0
        for i in range(n_vectors):
            x = X[i]
            norm = np.linalg.norm(x)
            x_unit = x / norm if norm > 0 else x

            # Rotate
            rx = apply_vilenkin_rotation(x_unit, signs1, signs2)
            # Quantize
            indices = nearest_centroid_indices(rx, centroids)
            rx_hat = centroids[indices]
            # Inverse rotate
            x_hat = apply_vilenkin_rotation_transpose(rx_hat, signs1, signs2) * norm

            mse = np.mean((x - x_hat) ** 2)
            total_mse += mse

        avg_mse = total_mse / n_vectors
        # Should be comparable to WHT-based PolarQuant
        # Paper bounds: MSE ≈ σ² × c(b) / d where c(b) depends on bit width
        assert avg_mse < 1.0, f"VHT MSE too high: {avg_mse:.4f} at {bit_width}-bit"

    @pytest.mark.parametrize("d", [80, 60, 96])
    def test_non_power_of_2_quantization(self, d):
        """VHT should enable quantization for dimensions WHT can't handle."""
        from turboquant.vilenkin import random_vilenkin_rotation, apply_vilenkin_rotation, apply_vilenkin_rotation_transpose
        from turboquant.codebook import optimal_centroids, nearest_centroid_indices

        bit_width = 3
        rng = np.random.default_rng(42)
        signs1, signs2 = random_vilenkin_rotation(d, rng)
        centroids = optimal_centroids(bit_width, d)

        x = rng.standard_normal(d)
        norm = np.linalg.norm(x)
        x_unit = x / norm

        rx = apply_vilenkin_rotation(x_unit, signs1, signs2)
        indices = nearest_centroid_indices(rx, centroids)
        rx_hat = centroids[indices]
        x_hat = apply_vilenkin_rotation_transpose(rx_hat, signs1, signs2) * norm

        mse = np.mean((x - x_hat) ** 2)
        # Just verify it works and produces reasonable results
        assert mse < 5.0, f"VHT quantization MSE too high for d={d}: {mse:.4f}"
        assert x_hat.shape == x.shape

    def test_vht_vs_wht_quality_comparison(self):
        """Compare VHT and WHT quantization MSE on d=128 (both should work)."""
        from turboquant.vilenkin import random_vilenkin_rotation, apply_vilenkin_rotation, apply_vilenkin_rotation_transpose
        from turboquant.rotation import apply_fast_rotation, apply_fast_rotation_transpose
        from turboquant.codebook import optimal_centroids, nearest_centroid_indices

        d = 128
        bit_width = 3
        n_vectors = 200

        centroids = optimal_centroids(bit_width, d)

        # VHT path
        rng_v = np.random.default_rng(42)
        s1_v, s2_v = random_vilenkin_rotation(d, rng_v)

        # WHT path (same seed for fair comparison)
        rng_w = np.random.default_rng(42)
        s1_w = rng_w.choice([-1.0, 1.0], size=d)
        s2_w = rng_w.choice([-1.0, 1.0], size=d)

        X = np.random.default_rng(99).standard_normal((n_vectors, d))

        mse_vht = 0.0
        mse_wht = 0.0
        for i in range(n_vectors):
            x = X[i]
            norm = np.linalg.norm(x)
            xu = x / norm if norm > 0 else x

            # VHT
            rv = apply_vilenkin_rotation(xu, s1_v, s2_v)
            idx_v = nearest_centroid_indices(rv, centroids)
            rv_hat = centroids[idx_v]
            xv_hat = apply_vilenkin_rotation_transpose(rv_hat, s1_v, s2_v) * norm
            mse_vht += np.mean((x - xv_hat) ** 2)

            # WHT
            rw = apply_fast_rotation(xu, s1_w, s2_w, d)
            idx_w = nearest_centroid_indices(rw, centroids)
            rw_hat = centroids[idx_w]
            xw_hat = apply_fast_rotation_transpose(rw_hat, s1_w, s2_w, d) * norm
            mse_wht += np.mean((x - xw_hat) ** 2)

        mse_vht /= n_vectors
        mse_wht /= n_vectors

        # For d=128 (power-of-2), VHT and WHT should give same results
        # since VHT reduces to WHT. Allow small numerical difference.
        np.testing.assert_allclose(mse_vht, mse_wht, rtol=0.05,
            err_msg=f"VHT MSE ({mse_vht:.6f}) != WHT MSE ({mse_wht:.6f}) for d=128")
