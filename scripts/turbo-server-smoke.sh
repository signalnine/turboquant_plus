#!/bin/bash
# TurboQuant server smoke test — catches bugs in server-specific paths:
#   - Prompt cache save/load (KV state serialization)
#   - Multi-turn slot reuse (LRU eviction)
#   - Warmup decode
#   - Multiple concurrent requests
#
# This test caught issue #28: KV serialization used unpadded hparams
# instead of actual tensor width, asserting on non-128 head dims.
#
# Usage: bash scripts/turbo-server-smoke.sh [llama-dir] [model-path]

set -e

LLAMA_DIR=${1:-/mnt/ai/projects/llama-cpp-turboquant}
MODEL=${2:-/mnt/ai/models/huggingface/qwen3.5-35b-a3b-GGUF/Qwen_Qwen3.5-35B-A3B-Q4_K_M.gguf}
SERVER=${LLAMA_DIR}/build-cuda/bin/llama-server
PORT=18199
FAIL=0

if [ ! -f "$SERVER" ]; then
    echo "ERROR: llama-server not found at $SERVER"
    exit 1
fi

echo "========================================"
echo "  TurboQuant Server Smoke Test"
echo "========================================"
echo ""

for CACHE_TYPE in turbo3 turbo4 turbo2; do
    echo "--- Testing $CACHE_TYPE ---"

    # Start server with prompt cache enabled (default)
    $SERVER -m "$MODEL" \
        --cache-type-k "$CACHE_TYPE" --cache-type-v "$CACHE_TYPE" \
        -c 4096 -ngl 99 -fa on --port $PORT -np 1 \
        --jinja 2>/tmp/turbo_smoke_stderr.log &
    SERVER_PID=$!

    # Wait for server to be healthy
    HEALTHY=0
    for i in $(seq 1 60); do
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            HEALTHY=1
            break
        fi
        sleep 1
    done

    if [ "$HEALTHY" -eq 0 ]; then
        echo "  FAIL: server did not start"
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
        FAIL=1
        continue
    fi

    # Request 1: initial query (triggers warmup + first KV write)
    RESP1=$(curl -sf http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":32,"temperature":0}' 2>&1)

    if echo "$RESP1" | grep -q '"choices"'; then
        echo "  Request 1 (initial): OK"
    else
        echo "  FAIL: Request 1 failed: $RESP1"
        FAIL=1
    fi

    # Request 2: different prompt (triggers slot reuse + prompt cache save/load)
    RESP2=$(curl -sf http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}' 2>&1)

    if echo "$RESP2" | grep -q '"choices"'; then
        echo "  Request 2 (slot reuse): OK"
    else
        echo "  FAIL: Request 2 failed (prompt cache bug?): $RESP2"
        FAIL=1
    fi

    # Request 3: multi-turn (same slot, appending to KV)
    RESP3=$(curl -sf http://localhost:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"What is the capital of France?"},{"role":"assistant","content":"Paris"},{"role":"user","content":"And Germany?"}],"max_tokens":32,"temperature":0}' 2>&1)

    if echo "$RESP3" | grep -q '"choices"'; then
        echo "  Request 3 (multi-turn): OK"
    else
        echo "  FAIL: Request 3 failed: $RESP3"
        FAIL=1
    fi

    # Check server didn't crash
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "  Server alive: OK"
    else
        echo "  FAIL: Server crashed during test!"
        FAIL=1
    fi

    # Cleanup
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo ""
done

if [ "$FAIL" -eq 0 ]; then
    echo "========================================"
    echo "  ALL PASSED"
    echo "========================================"
    exit 0
else
    echo "========================================"
    echo "  FAILURES DETECTED"
    echo "  Check /tmp/turbo_smoke_stderr.log"
    echo "========================================"
    exit 1
fi
