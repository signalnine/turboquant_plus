"""Validate Vilenkin-Hartley rotation on real KV cache data.

Captures KV cache tensors from actual model forward passes, then compares
TurboQuant compression quality using dense, WHT, and Vilenkin rotations.

Tests two models:
  - Qwen2.5-1.5B (head_dim=128, power-of-2 baseline)
  - Qwen3-4B (head_dim=80, non-power-of-2 motivating case)

Metrics:
  - MSE: reconstruction error per element
  - Cosine similarity: attention score fidelity
  - Top-1 match rate: does compressed K produce the same argmax as original?
"""

import torch
import numpy as np
import sys
import os

# Ensure turboquant is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def capture_kv_cache(model_name: str, prompt: str = "The capital of France is",
                     max_new_tokens: int = 1, device: str = "cuda"):
    """Run a forward pass and capture the KV cache tensors."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    past_kv = outputs.past_key_values
    K_tensors = []
    V_tensors = []

    if hasattr(past_kv, 'layers'):
        # DynamicCache with .layers (transformers >= 5.x)
        for layer in past_kv.layers:
            k = layer.keys[0].cpu().float().numpy()   # (n_kv_heads, seq_len, head_dim)
            v = layer.values[0].cpu().float().numpy()
            K_tensors.append(k)
            V_tensors.append(v)
    elif hasattr(past_kv, 'key_cache'):
        # DynamicCache with key_cache (transformers 4.38-4.x)
        for i in range(len(past_kv.key_cache)):
            k = past_kv.key_cache[i].squeeze(0).cpu().float().numpy()
            v = past_kv.value_cache[i].squeeze(0).cpu().float().numpy()
            K_tensors.append(k)
            V_tensors.append(v)
    else:
        # Tuple-style (transformers < 4.38)
        for i in range(len(past_kv)):
            k = past_kv[i][0].squeeze(0).cpu().float().numpy()
            v = past_kv[i][1].squeeze(0).cpu().float().numpy()
            K_tensors.append(k)
            V_tensors.append(v)

    n_layers = len(K_tensors)

    head_dim = K_tensors[0].shape[-1]
    n_kv_heads = K_tensors[0].shape[0]
    seq_len = K_tensors[0].shape[1]

    print(f"  Captured: {n_layers} layers, {n_kv_heads} KV heads, seq_len={seq_len}, head_dim={head_dim}")

    del model, tokenizer
    torch.cuda.empty_cache()

    return K_tensors, V_tensors, head_dim, n_layers, n_kv_heads, seq_len


def compress_and_measure(vectors, head_dim, bit_width, rotation_method):
    """Compress a set of vectors and measure reconstruction quality.

    Args:
        vectors: list of arrays, each shape (n_heads, seq_len, head_dim)
        head_dim: dimension of each vector
        bit_width: quantization bits
        rotation_method: 'dense', 'wht', or 'vilenkin'

    Returns:
        dict with mse, cosine_sim, and timing info
    """
    from turboquant.polar_quant import PolarQuant

    # Skip WHT for non-power-of-2
    if rotation_method == 'wht' and (head_dim & (head_dim - 1)) != 0:
        return None

    pq = PolarQuant(head_dim, bit_width, seed=42, rotation_method=rotation_method)

    total_mse = 0.0
    total_cosine = 0.0
    n_vectors = 0

    for layer_vecs in vectors:
        n_heads, seq_len, d = layer_vecs.shape
        for h in range(n_heads):
            for s in range(seq_len):
                x = layer_vecs[h, s]
                idx, norm = pq.quantize(x)
                x_hat = pq.dequantize(idx, norm)

                mse = np.mean((x - x_hat) ** 2)
                cos = np.dot(x, x_hat) / (np.linalg.norm(x) * np.linalg.norm(x_hat) + 1e-10)

                total_mse += mse
                total_cosine += cos
                n_vectors += 1

    return {
        'mse': total_mse / n_vectors,
        'cosine': total_cosine / n_vectors,
        'n_vectors': n_vectors,
    }


def measure_attention_fidelity(K_tensors, V_tensors, head_dim, bit_width, rotation_method):
    """Measure how well compressed K preserves attention scores.

    For each layer, compute Q×K^T with original and compressed K,
    then measure cosine similarity of attention distributions and top-1 match rate.
    """
    from turboquant.polar_quant import PolarQuant

    if rotation_method == 'wht' and (head_dim & (head_dim - 1)) != 0:
        return None

    pq = PolarQuant(head_dim, bit_width, seed=42, rotation_method=rotation_method)

    total_attn_cosine = 0.0
    total_top1_match = 0.0
    n_samples = 0

    for layer_idx in range(len(K_tensors)):
        K = K_tensors[layer_idx]  # (n_heads, seq_len, head_dim)
        n_heads, seq_len, d = K.shape

        for h in range(n_heads):
            # Compress all K vectors for this head
            K_hat = np.zeros_like(K[h])
            for s in range(seq_len):
                idx, norm = pq.quantize(K[h, s])
                K_hat[s] = pq.dequantize(idx, norm)

            # Use last token as query (simulate decode attention)
            q = K[h, -1]  # use last K as proxy for Q (same distribution)

            # Attention scores: Q × K^T
            scores_orig = q @ K[h].T  # (seq_len,)
            scores_hat = q @ K_hat.T

            # Softmax
            def softmax(x):
                e = np.exp(x - np.max(x))
                return e / e.sum()

            attn_orig = softmax(scores_orig / np.sqrt(d))
            attn_hat = softmax(scores_hat / np.sqrt(d))

            # Cosine similarity of attention distributions
            cos = np.dot(attn_orig, attn_hat) / (np.linalg.norm(attn_orig) * np.linalg.norm(attn_hat) + 1e-10)
            total_attn_cosine += cos

            # Top-1 match
            top1_match = 1.0 if np.argmax(attn_orig) == np.argmax(attn_hat) else 0.0
            total_top1_match += top1_match

            n_samples += 1

    return {
        'attn_cosine': total_attn_cosine / n_samples,
        'top1_match': total_top1_match / n_samples,
        'n_samples': n_samples,
    }


def run_model_benchmark(model_name, prompt="The quick brown fox jumps over the lazy dog. " * 10):
    """Run full benchmark for one model."""
    K_tensors, V_tensors, head_dim, n_layers, n_kv_heads, seq_len = capture_kv_cache(model_name, prompt)

    is_pow2 = (head_dim & (head_dim - 1)) == 0
    methods = ['dense', 'vilenkin']
    if is_pow2:
        methods.append('wht')

    print(f"\n{'='*80}")
    print(f"Model: {model_name} | head_dim={head_dim} | {n_layers} layers | {n_kv_heads} KV heads | seq_len={seq_len}")
    print(f"{'='*80}")

    for bw in [3, 4]:
        print(f"\n--- {bw}-bit ---")
        print(f"{'Method':>10} | {'K MSE':>10} {'V MSE':>10} {'K cos':>8} {'V cos':>8} | {'Attn cos':>9} {'Top-1':>7}")
        print(f"{'-'*10}-+-{'-'*10}-{'-'*10}-{'-'*8}-{'-'*8}-+-{'-'*9}-{'-'*7}")

        for method in methods:
            k_res = compress_and_measure(K_tensors, head_dim, bw, method)
            v_res = compress_and_measure(V_tensors, head_dim, bw, method)
            attn_res = measure_attention_fidelity(K_tensors, V_tensors, head_dim, bw, method)

            if k_res is None:
                continue

            print(f"{method:>10} | {k_res['mse']:>10.6f} {v_res['mse']:>10.6f} "
                  f"{k_res['cosine']:>8.5f} {v_res['cosine']:>8.5f} | "
                  f"{attn_res['attn_cosine']:>9.6f} {attn_res['top1_match']:>7.1%}")


if __name__ == "__main__":
    # Longer prompt for more KV cache entries
    prompt = ("The history of artificial intelligence began in antiquity, with myths and stories "
              "of artificial beings endowed with intelligence. The seeds of modern AI were planted "
              "by philosophers who attempted to describe the process of human thinking as the "
              "mechanical manipulation of symbols. This work culminated in the invention of the "
              "programmable digital computer in the 1940s, a machine based on the abstract essence "
              "of mathematical reasoning. This device and the ideas behind it inspired a handful of "
              "scientists to begin seriously discussing the possibility of building an electronic brain.")

    print("Vilenkin-Hartley Validation on Real KV Cache Data")
    print("=" * 80)

    # Model 1: head_dim=128 (power-of-2 baseline)
    run_model_benchmark("Qwen/Qwen2.5-1.5B", prompt)

    # Model 2: head_dim=80 (non-power-of-2, the motivating case)
    run_model_benchmark("Qwen/Qwen3-4B", prompt)
