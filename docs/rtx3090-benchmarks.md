# TurboQuant CUDA — RTX 3090 Benchmarks (24 GB)

Hardware: RTX 3090 (24 GB, SM 8.6 Ampere)
Model: Qwen2.5-7B-Instruct Q2_K (2.9 GB)
Date: 2026-03-29

---

## Decode Speed (llama-bench pp512/tg128, 3 runs)

| Type | pp512 t/s | tg128 t/s | vs q8_0 |
|------|----------|----------|---------|
| f16 | 4,266 | 148 | 1.04× |
| q8_0 | 4,241 | 142 | baseline |
| turbo4 | 4,030 | 134 | 0.94× |
| turbo3 | 4,052 | 138 | 0.97× |
| turbo2 | 4,057 | 136 | 0.96× |

All turbo types within 3-6% of q8_0 decode speed on SM 8.6.
No Ampere-specific regressions vs SM 12.0 (Blackwell).

Earlier reports of turbo2 dropping to 63 t/s at 32K were measurement
noise from variable-length generation with `--single-turn`. Fixed
`llama-bench` (tg128, 3 runs) shows stable 136 t/s at all contexts.

## VRAM Usage at 128K Context

| Type | Model | KV cache | Compute | Total used | Free |
|------|-------|----------|---------|-----------|------|
| turbo3 | 2,700 | **1,568** | 398 | 4,666 | 19,145 |
| q8_0 | 2,700 | **3,808** | 412 | 6,920 | 16,895 |

turbo3 saves **2,240 MiB** of KV cache vs q8_0 at 128K context (2.4× compression).

## Key Findings

**All turbo types match q8_0 speed on Ampere:**
- turbo3: 97% of q8_0 decode (138 vs 142 t/s)
- turbo2: 96% (136 t/s)
- turbo4: 94% (134 t/s)
- No SM 8.6-specific issues — same relative performance as SM 12.0

**VRAM savings are the real win on 24GB cards:**
- turbo3 at 128K: 1.6 GB KV vs 3.8 GB for q8_0
- Saves 2.2 GB — enough for longer context or larger model
- Tested up to 256K on 24GB with no OOM

## No OOM Issues (issue #14)

Previously reported OOM at 64K was from the old Metal-first branch.
Our CUDA port has no "shadow buffer" — VEC FA dequantizes turbo3
inline with no temporary f16 allocation.
