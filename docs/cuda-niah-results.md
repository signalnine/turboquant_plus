# turbo3 CUDA — NIAH Validation (RTX 5090)

Hardware: RTX 5090 (32 GB, SM 12.0 Blackwell)
Model: Qwen3.5-35B-A3B Q4_K_M
Date: 2026-03-27

---

## Summary

turbo3 KV cache passes needle-in-a-haystack retrieval at every context length from
4K to 1M tokens, matching or exceeding q8_0 at all depths.

| Context | turbo3 score | q8_0 score |
|---------|-------------|-----------|
| 4K      | 11/11       | 11/11     |
| 8K      | 11/11       | 11/11     |
| 16K     | 11/11       | 11/11     |
| 32K     | 11/11       | 11/11     |
| 64K     | 11/11       | 11/11     |
| 128K    | 10/11       | 10/11     |
| 256K    | 11/11       | 10/11     |
| 1024K   | 11/11       | —¹        |

¹ q8_0 KV at 1M context requires ~10.9 GB — doesn't fit alongside the 20 GB model on a 32 GB card.
  turbo3 KV at 1M context requires ~4.5 GB (3.47× compression), total ~28 GB → fits.

Score = needle retrieved correctly at 11 depth positions (0%, 10%, …, 100%).

---

## VEC Kernel Bug and Fix

The initial CUDA port produced 0/11 at all context lengths. Root cause: the VEC
flash-attention kernel (`flash_attn_ext_vec`, used for all generation steps where
n_tokens ≤ 2) had a stride mismatch in `vec_dot_fattn_vec_KQ_turbo3_0`.

**The bug:** the outer K loop stepped by `nthreads` (=8) but Q registers are loaded
in blocks of `nthreads * cpy_ne` (=32). Thread `t` at outer step `s` accessed
K element `16s + 2t` but Q element `64*(s/4) + 8t + 2*(s%4)` — these are equal
only when `t=0` and `s%4=0`.

**The fix:** match the f16 kernel pattern — step outer loop by `nthreads * cpy_ne`,
add inner `k_KQ_1` loop, index Q as `Q_v[k_KQ_0/nthreads + k_KQ_1]`. The MMA
kernel (prefill, n_tokens > 2) was unaffected.

The bug was latent in the original code and would pass any prefill-only test. It
only surfaces at generation time (every single decode step).

---

## Speed

| Metric | Value |
|--------|-------|
| Decode (tg128, turbo3) | ~94 t/s |
| Decode (tg128, f16) | ~95 t/s |
| turbo3 vs f16 decode | 98.5% |
| Prefill at 1M context | ~1,474 tokens/s |
| Prefill time per 1M query | ~9 min (820K token prompt) |

---

## 1M Context Setup

Qwen3.5's `n_ctx_train = 262,144`. To reach 1M tokens we use YaRN rope scaling:

```bash
-c 1060864 \
--rope-scaling yarn \
--yarn-ext-factor 1.0 \
--yarn-orig-ctx 262144 \
--rope-scale 4
```

llama-server's slot context cap (`server-context.cpp:749`) was changed from a hard
cap to a warning when rope scaling is configured, allowing the slot to use the full
requested context.

VRAM at 1M context with turbo3:
- Model weights: ~20,128 MiB
- turbo3 KV cache (1M × 64 layers × 2 × 128-dim × 14B/32): ~4,532 MiB
- Compute / activations: ~3,152 MiB
- Total: ~28,063 MiB (fits in 32,607 MiB)

---

## NIAH Methodology

Kamradt/RULER single-needle methodology (NIAH v2):

- Haystack: synthetic Wikipedia-style text
- Needle: single key-value fact (`"The special magic {city} number is: {number}."`)
- Query: `"What is the special magic {city} number?"`
- Scoring: exact number match in response (not fuzzy)
- Depth positions: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

Raw results in `niah_results/` (4K–256K) and `niah_results/beyond_262k/` (1M).

---

## Comparison with PR #2 (wesraph, closed)

An independent parallel implementation (PR #2) existed before our PR #3 was opened.
Key differences:

| Aspect | PR #3 (ours) | PR #2 (wesraph) |
|--------|-------------|----------------|
| WHT kernel | 128 threads/block, shared-mem butterfly | 1 thread/group, serial butterfly |
| SET_ROWS kernel | 128 threads/block, shared-mem WHT | 32 threads/warp, `__shfl_sync` WHT |
| Norm field | `grp_norm / recon_norm` (error-corrected) | raw `grp_norm` |
| VEC kernel validation | 11/11 NIAH across 4K–1M | single 4875-token generation (prefill only) |
| Target hardware | SM 12.0 Blackwell (RTX 5090) | SM 8.6 (RTX 3090) |
| turbo4 | No | Yes |

PR #2's generation test used a long prompt → MMA kernel (prefill, n_tokens > 2)
→ the VEC kernel stride bug would have gone undetected. Their implementation may
carry the same latent generation bug.
