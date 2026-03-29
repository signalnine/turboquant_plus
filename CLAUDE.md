# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TurboQuant+ implements the TurboQuant paper (ICLR 2026, arXiv 2504.19874) — KV cache compression for LLM inference. It achieves 4.6x compression of transformer KV caches with q8_0 speed parity on Apple Silicon and ~1% quality loss.

Two-stage algorithm:
1. **PolarQuant** (Algorithm 1): Random rotation + optimal scalar quantization (b-1 bits)
2. **QJL**: 1-bit residual correction preserving inner products

There is a companion llama.cpp fork (separate repo) with C port and Metal GPU kernels.

## Build & Test Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests (141 tests, CI enforces >= 95% coverage)
pytest tests/ -v --cov=turboquant --cov-fail-under=95

# Run single test file
pytest tests/test_turboquant.py -v

# Run specific test
pytest tests/test_turboquant.py::TestTurboQuantRoundTrip::test_mse_within_paper_bounds -v

# Quick demo (no model needed)
python benchmarks/demo.py

# Pre-push quality gate
bash scripts/turbo-quality-gate.sh

# Server smoke test (prompt cache, multi-turn, slot reuse)
bash scripts/turbo-server-smoke.sh [llama-dir] [model-path]
```

Dependencies: Python >= 3.10, numpy >= 1.24, scipy >= 1.10. Optional: `matplotlib` (bench), `torch`/`transformers`/`accelerate` (real model validation).

## Architecture

### Core modules (`turboquant/`)

- **`turboquant.py`** — `TurboQuant` (Algorithm 2: PolarQuant + QJL) and `TurboQuantMSE` (MSE-only, no QJL stage). TurboQuant used for K cache (preserves inner products), TurboQuantMSE for V cache (preserves MSE).
- **`polar_quant.py`** — `PolarQuant` class: rotation + scalar quantization. Extracts L2 norms before quantizing (critical for non-unit-norm KV tensors).
- **`qjl.py`** — `QJL` class: 1-bit Quantized Johnson-Lindenstrauss residual correction.
- **`rotation.py`** — `random_rotation_dense()` (Haar via QR, exact) and `random_rotation_fast()` (D1 @ Hadamard @ D2, O(d log d)). Fast variant used in Metal kernels.
- **`codebook.py`** — `optimal_centroids()`: 1-bit/2-bit closed-form, 3+ bit via Lloyd's algorithm on Gaussian N(0, 1/d). Uses `searchsorted` for O(n log k) nearest centroid.
- **`kv_cache.py`** — `KVCacheCompressor`: compresses full (num_layers, num_heads, seq_len, head_dim) KV caches.
- **`outlier.py`** — Non-integer bit-widths (2.5-bit, 3.5-bit) via outlier channel splitting.
- **`utils.py`** — Bit packing (`pack_bits`, `pack_indices`), memory footprint calculations.

### Key patterns

- All quantize functions handle both 1D (single vector) and 2D (batch) inputs, unwrapping at the end if single.
- L2 norms are extracted and stored in float before quantization, then used to rescale on dequantization.
- All randomness uses seeded `np.random.default_rng(seed)` — never global `np.random`.
- Vectorized NumPy operations throughout; avoid Python loops over coordinates.

### Tests (`tests/`)

Tests are parametrized across bit-widths (2, 3, 4) and dimensions (64, 128, 256, 1536, 3072). Validation tests check paper's theoretical MSE/distortion bounds.

### Scripts

- `scripts/niah_test.py` — Needle-in-a-haystack benchmark (NIAH v2, Kamradt/RULER methodology)
- `scripts/turbo-quality-gate.sh` — Pre-push gate checking PPL + context scaling
- `scripts/turbo_hardware_diag.py` — Hardware diagnostics for M5 Max profiling

## CI

GitHub Actions (`.github/workflows/ci.yml`): runs on push/PR to main, matrix of Python 3.10/3.11/3.12, enforces 95% coverage.
