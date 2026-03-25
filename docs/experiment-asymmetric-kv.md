# Experiment: Asymmetric K/V Compression

Branch: `experiment/asymmetric-kv`

## Hypothesis
V carries content (needs precision), K carries direction (needs less). Higher bits for V, lower for K, same total bits = better quality.

## Test Results (8 chunks, wikitext-2, Qwen3.5-35B-A3B)

| K cache | V cache | PPL | Speed | Status |
|---------|---------|-----|-------|--------|
| turbo3 | turbo3 | 6.19 | 4833 | baseline |
| turbo3 | q8_0 | — | — | CRASHED: no mixed-type FA kernel |
| q8_0 | turbo3 | 161.6 | 2023 | BROKEN: rot tensors not allocated when K is non-turbo |
| turbo3 | q4_0 | — | — | CRASHED: no mixed-type FA kernel |
| q4_0 | turbo3 | 156.2 | 2049 | BROKEN: same rot tensor issue |
| q8_0 | q8_0 | 6.11 | 4958 | reference |

## Blockers Identified

1. **No mixed-type flash attention kernels.** The template supports different K/V types (`block_k` vs `block_v`) but only same-type instantiations exist. Need new template instantiations for each combo.

2. **Rotation tensors gated on K type only.** Allocated when `type_k == turbo3/4` but not when only `type_v` is turbo. Fix: check both types.

3. **V un-rotation gate fixed.** Changed from `k->type` to `v->type` check (Codex had flagged this).

## Engineering Required

- Add flash attention kernel instantiations for turbo3/q8_0 and turbo3/q4_0 combinations
- Fix rotation tensor allocation to check `type_k OR type_v`
- Add arg validation: if K is turbo but V is not, warn about V rotation mismatch
- Test both directions (turbo K + standard V, standard K + turbo V)

## Verdict

**Not a quick win.** Requires significant flash attention template work. Parking for now. The uniform turbo3/turbo3 config already achieves q8_0 parity — asymmetric would only help if we want better quality at the same compression or more compression at the same quality.
