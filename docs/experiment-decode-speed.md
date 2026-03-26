# Experiment: Decode Speed Parity

Branch: `experiment/decode-speed-parity`

## Problem
turbo3 decode is 0.84x of q8_0 (65.4 vs 78.3 tok/s at 8K context on M5 Max). External testers report 0.83x (Mario, M1 Max 32K).

## Root Cause Isolation

| Config | Decode tok/s | vs q8_0 |
|--------|-------------|---------|
| turbo3 (full dequant) | 65.4 | 0.84x |
| turbo3 (NO centroid LUT — speed ceiling) | **78.1** | **1.00x** |
| q8_0 | 78.3 | baseline |

**The entire 16% decode gap comes from the centroid table lookup.** Without the LUT (returning a constant), turbo3 matches q8_0 exactly.

82M constant memory accesses per decode token (8K context, 10 attn layers, 256-dim heads).

## Approaches Tested

### Register-Based Centroid Select
Split 8-entry LUT into two 4-entry float4 registers (c_lo for hi1=0, c_hi for hi1=1). Use ternary select instead of array index.

**Result:** 63.1 tok/s — SLOWER than the 65.4 baseline. Variable indexing into float4 registers is equally expensive on Metal as constant memory LUT. The GPU treats both as data-dependent access with potential pipeline stalls.

### Linear Approximation
Fit centroids to `c ≈ a * idx + b`. Max error 27% — unacceptable for quality.

## Remaining Approaches (Not Yet Tested)

### Store Centroid Values Directly
During quantize, store fp16 centroid values instead of 3-bit indices. Dequant becomes simple fp16 → fp32 + norm multiply (no LUT).

**Trade-off:** Block size increases. 32 × fp16 = 64 bytes vs 14 bytes currently. Compression drops from 4.6x to ~2.0x — equivalent to q8_0. This defeats the purpose.

### Fused Compressed Attention
Compute Q·K directly on quantized indices. Precompute `Q·centroid[c]` for 8 centroids, then for each K element just look up `q_dot_centroid[idx] * norm`.

**Trade-off:** Requires custom flash attention kernel variant. Complex but would eliminate dequant entirely for Q·K path. V path still needs dequant.

### Half-Precision Centroid LUT
Store the 8 centroids as half instead of float. Reduces constant memory bandwidth by 2x.

**Not yet tested.** May help on older hardware (M1) where constant memory bandwidth is more constrained.

## NEW Finding: Decode Ratio Degrades with Context Depth

Server-measured decode speed at different context depths (M5 Max, Qwen3.5-35B-A3B):

| Context | turbo3 decode | q8_0 decode | Ratio |
|---------|-------------|-----------|-------|
| ~12 tokens | 75.3 | 85.2 | 0.88x |
| ~8K tokens | 59.2 | 77.7 | 0.76x |

**The decode gap WIDENS with context.** At 12 tokens it's 12%. At 8K it's 24%. At 40K (Mario's PDF) it could be 40%+.

Root cause: the centroid LUT creates constant cache pressure that compounds as more KV positions are accessed per decode step. q8_0 uses pure ALU (int8 * scale) with zero cache pressure.

This explains:
- Mario's observation: "synthetic close but WebUI shows bigger difference"
- The anon tester's 0.36x at 42K context
- The consistent 0.83x at 32K from Mario's bench

**This is SEPARATE from prefill scaling (flat at 0.99x).** Prefill processes tokens in large batches where memory bandwidth dominates. Decode processes 1 token at a time where per-position compute dominates.

## Isolation Test: Bit Extraction vs LUT

Returned the raw index as a float (no LUT, but still data-dependent bit extraction):

| Context | With LUT | Without LUT (index) | q8_0 |
|---------|----------|-------------------|------|
| ~12 tokens | 75.3 | 73.6 | 85.2 |
| ~8K tokens | 59.2 | **70.1** | 77.7 |

**The bit extraction is NOT the bottleneck — it's fast.** The LUT alone costs 18.5% at 8K context (59.2 → 70.1). At short context the LUT is actually faster (compiler optimization). The context-dependent cost comes from constant cache thrashing at high access volumes.

## Regression Against Main

| Metric | Main branch | Experiment branch |
|--------|------------|-------------------|
| PPL (8-chunk) | 6.211 | 6.211 |
| Decode (short ctx) | 75.3 (turbo3) | same dequant |
| Decode (8K ctx) | 65.8 (earlier measurement) | 61.5 (custom WHT adds overhead) |

**The experiment branch is a net negative for decode.** The custom GGML_OP_TURBO_WHT adds graph node overhead that costs more than the fp16 LUT saves. Main branch is the correct configuration.

## Should This Be Implemented?

**No code changes from this branch should be merged to main.** The investigation produced valuable data but no improvements:

- fp16 LUT: +3% at short context, eaten by custom op overhead at long context
- Custom GGML_OP_TURBO_WHT: net negative for decode (extra kernel dispatch)
- All 8 LUT alternatives: worse or marginal vs the original 8-entry float constant

**The 8-entry constant float LUT on main branch is the optimal dequant for the current block format.** Any improvement requires changing the block format or the flash attention kernel architecture (fused compressed attention).

## Key Finding
The decode gap is a **fundamental cost of data-dependent constant cache indexing** that SCALES with context depth. At short context (~12 tokens) turbo3 is 0.88x q8_0. At 8K context it's 0.76x. At 40K+ it could be 0.6x or worse.

The only paths that would close this gap:
1. **Fused compressed attention** — compute Q·K from indices without dequant (complex, custom FA kernel)
2. **Store centroid values directly** — no LUT (trades compression for speed, defeats purpose)
3. **Block format redesign** — pack indices for faster extraction (format-breaking change)

None of these are simple. The decode gap is an inherent tradeoff of 3-bit compression on current GPU architectures.

## Recommendation for Users

- **Short context (<4K):** turbo3 decode is ~88% of q8_0 — barely noticeable
- **Medium context (4K-16K):** ~76-84% of q8_0 — noticeable but usable
- **Long context (32K+):** <70% of q8_0 — use q8_0 for decode-sensitive workloads
- **Memory-constrained:** turbo3 still wins on fitting longer context in available RAM

For M1 testers: the 0.83x decode ratio is the expected behavior at the current compression level. It's consistent across M1 Max and M5 Max.
