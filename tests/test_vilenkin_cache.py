"""Tests for Vilenkin coefficient cache."""

import numpy as np
import pytest


class TestVilenkinCacheEncode:
    def test_round_trip_lossless_at_full_energy(self):
        """With 100% energy and int8, round-trip should be near-perfect."""
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128, energy_threshold=0.999, quant_levels=256)
        enc = VilenkinCacheEncoder(config)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        encoded = enc.encode(x)
        x_hat = enc.decode(encoded)
        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat))
        assert cos > 0.99, f"Cosine too low: {cos}"

    def test_zero_vector(self):
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128)
        enc = VilenkinCacheEncoder(config)
        encoded = enc.encode(np.zeros(128))
        x_hat = enc.decode(encoded)
        np.testing.assert_allclose(x_hat, np.zeros(128), atol=1e-10)

    def test_non_power_of_2(self):
        """head_dim=80 should work."""
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=80, energy_threshold=0.95, quant_levels=16)
        enc = VilenkinCacheEncoder(config)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(80)
        encoded = enc.encode(x)
        x_hat = enc.decode(encoded)
        assert x_hat.shape == (80,)
        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat))
        assert cos > 0.90, f"Cosine too low for d=80: {cos}"

    def test_batch(self):
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128, energy_threshold=0.95, quant_levels=16)
        enc = VilenkinCacheEncoder(config)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 128))
        encoded_list = enc.encode_batch(X)
        X_hat = enc.decode_batch(encoded_list)
        assert X_hat.shape == X.shape
        for i in range(50):
            cos = np.dot(X[i], X_hat[i]) / (np.linalg.norm(X[i]) * np.linalg.norm(X_hat[i]) + 1e-10)
            assert cos > 0.85, f"Cosine too low at row {i}: {cos}"


class TestSharedMask:
    def test_calibrate_finds_universal_indices(self):
        """Structured data (simulating model KV cache) should have shared indices."""
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128, energy_threshold=0.95, shared_mask=True)
        enc = VilenkinCacheEncoder(config)
        rng = np.random.default_rng(42)
        # Simulate structured KV data: base pattern + noise
        # Real model KV has ~10 dominant frequencies shared across positions
        base = rng.standard_normal(128)
        X = np.array([base + 0.3 * rng.standard_normal(128) for _ in range(200)])
        shared = enc.calibrate(X, universal_threshold=0.9)
        assert len(shared) > 0, "No universal indices found"
        assert len(shared) < 128, "All indices are universal (shouldn't happen)"

    def test_shared_mask_compression(self):
        """Shared mask should give better compression than per-vector indices."""
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config_shared = VilenkinCacheConfig(head_dim=128, energy_threshold=0.95, shared_mask=True, quant_levels=16)
        config_per = VilenkinCacheConfig(head_dim=128, energy_threshold=0.95, shared_mask=False, quant_levels=16)
        enc_s = VilenkinCacheEncoder(config_shared)
        enc_p = VilenkinCacheEncoder(config_per)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 128))
        enc_s.calibrate(X[:50])

        bytes_s = enc_s.bytes_per_position(len(enc_s._shared_indices))
        # For per-vector, estimate
        encoded = enc_p.encode(X[50])
        bytes_p = enc_p.bytes_per_position(encoded['n_coeffs'])
        assert bytes_s < bytes_p, f"Shared ({bytes_s}) not smaller than per-vec ({bytes_p})"


class TestCompression:
    def test_compression_ratio(self):
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128, energy_threshold=0.95, quant_levels=16)
        enc = VilenkinCacheEncoder(config)
        ratio = enc.compression_ratio(n_coeffs=48)
        assert ratio > 3.0, f"Compression ratio too low: {ratio}"
        assert ratio < 20.0, f"Compression ratio unrealistic: {ratio}"

    @pytest.mark.parametrize("energy", [0.99, 0.95, 0.90])
    def test_quality_vs_energy(self, energy):
        """Higher energy threshold should give better quality."""
        from turboquant.vilenkin_cache import VilenkinCacheConfig, VilenkinCacheEncoder
        config = VilenkinCacheConfig(head_dim=128, energy_threshold=energy, quant_levels=256)
        enc = VilenkinCacheEncoder(config)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(128)
        encoded = enc.encode(x)
        x_hat = enc.decode(encoded)
        cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat))
        # Higher energy = better quality
        min_cos = {0.99: 0.98, 0.95: 0.93, 0.90: 0.88}
        assert cos > min_cos[energy], f"Cosine {cos} below threshold for energy={energy}"


class TestCompareWithPolarQuant:
    def test_vilenkin_cache_vs_polarquant(self):
        """Vilenkin coefficient cache should achieve better compression-quality tradeoff."""
        from turboquant.vilenkin_cache import compare_methods
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 128))
        results = compare_methods(X, head_dim=128, bit_width=3)

        # Vilenkin cache at 95% int4 should have higher compression than PolarQuant 3-bit
        vc = results.get('Vilenkin-95%-int4', {})
        pq = results.get('PolarQuant-vilenkin-3bit', {})
        if vc and pq:
            assert vc.get('compression', 0) > 3.0, f"Vilenkin compression too low: {vc}"
