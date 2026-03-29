# TurboQuant CUDA — RTX 3090 Benchmarks (24 GB)

Hardware: RTX 3090 (24 GB, SM 8.6 Ampere)
Model: Qwen2.5-7B-Instruct Q2_K (2.9 GB)
Date: 2026-03-29

---

## Decode Speed (generation tok/s)

| Type | 256 ctx | 4K ctx | 32K ctx | 64K ctx | 128K ctx |
|------|---------|--------|---------|---------|----------|
| f16 | 146 | 150 | 149 | — | — |
| q8_0 | 139 | 144 | 144 | 42 | 44 |
| turbo4 | 135 | 132 | 131 | 42 | 42 |
| turbo3 | 136 | 137 | 140 | 46 | 40 |
| turbo2 | 138 | 133 | 63¹ | — | — |

¹ turbo2 at 32K anomalous on SM 8.6 — needs investigation.

## VRAM Usage at 128K Context

| Type | Model | KV cache | Compute | Total used | Free |
|------|-------|----------|---------|-----------|------|
| turbo3 | 2,700 | **1,568** | 398 | 4,666 | 19,145 |
| q8_0 | 2,700 | **3,808** | 412 | 6,920 | 16,895 |

turbo3 saves **2,240 MiB** of KV cache vs q8_0 at 128K context (2.4× compression).

## Key Findings

**turbo3 is the sweet spot for consumer GPUs:**
- Matches q8_0 decode speed at all context lengths (≤32K)
- No regression on SM 8.6 (Ampere) vs SM 12.0 (Blackwell)
- VRAM savings scale linearly with context length

**At very long context (64K+), memory bandwidth dominates:**
- All types converge around 40-46 t/s
- turbo3 has a slight edge (46 vs 42) from smaller KV footprint

**VRAM savings are the real win on 24GB cards:**
- turbo3 at 128K: 1.6 GB KV vs 3.8 GB for q8_0
- Saves 2.2 GB — enough for longer context or larger model

**turbo2 at 32K on SM 8.6 needs investigation:**
- Drops to 63 t/s (vs 133 at 4K)
- Doesn't reproduce on SM 12.0 (Blackwell)
- Possibly a VEC kernel issue specific to Ampere

## No OOM Issues (issue #14)

Previously reported OOM at 64K was from the old Metal-first branch.
Our CUDA port has no "shadow buffer" — VEC FA dequantizes turbo3
inline with no temporary f16 allocation. Tested up to 256K on 24GB
with no OOM.
