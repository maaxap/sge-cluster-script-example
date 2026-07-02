# Usage: python scripts/benchmark_vllm.py
import json
import subprocess
import time
from datetime import datetime, timezone

from vllm import LLM, SamplingParams


# MODEL_REPO  = "Qwen/Qwen3-32B-AWQ"
# QUANTIZATION = "awq_marlin"
# ATTENTION_BACKEND = "FLASH_ATTN"

MODEL_REPO  = "cyankiwi/gemma-4-31B-it-AWQ-4bit"
QUANTIZATION = None
ATTENTION_BACKEND = None # "TRITON_ATTN"

N_CTX       = 13312
MAX_TOKENS  = 13312
TEST_PROMPT = "Explain the theory of relativity in detail as long as you can."
TEMPERATURE = 1.0
BATCH_SIZE  = 8
N_BATCHES   = 4
TENSOR_PARALLEL_SIZE = 2
# TENSOR_PARALLEL_SIZE = 1


def get_gpu_memory():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().split("\n"):
        idx, name, used, total = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "used_mb": int(used), "total_mb": int(total)})
    return gpus


def load_model(tensor_parallel_size: int) -> LLM:
    return LLM(
        model=MODEL_REPO,
        download_dir=str("./hf_cache"),
        quantization=QUANTIZATION,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.95,
        max_model_len=N_CTX,
        enforce_eager=False,
        dtype="auto",
        enable_chunked_prefill=True,
        attention_backend=ATTENTION_BACKEND
    )


def run_benchmark():
    print("=" * 60)
    print(f"Model : {MODEL_REPO}")

    print("\nGPU Memory BEFORE load:")
    before = get_gpu_memory()
    for g in before:
        print(f"  GPU {g['index']} ({g['name']}): {g['used_mb']} / {g['total_mb']} MB")

    print("\nLoading model with vLLM...")
    llm = load_model(tensor_parallel_size=TENSOR_PARALLEL_SIZE)

    print(f"  tensor_parallel_size : {TENSOR_PARALLEL_SIZE}")

    print("\nGPU Memory AFTER load:")
    after = get_gpu_memory()
    for g in after:
        delta = g["used_mb"] - before[g["index"]]["used_mb"]
        print(f"  GPU {g['index']} ({g['name']}): {g['used_mb']} / {g['total_mb']} MB  (+{delta} MB)")

    print("\n" + "=" * 60)
    print("Warming up (triggers CUDA graph compilation)...")
    warmup_params = SamplingParams(max_tokens=20, temperature=TEMPERATURE)
    llm.generate([TEST_PROMPT] * BATCH_SIZE, warmup_params)
    print("  Warmup done.")

    print("\nTTFT proxy (max_tokens=1)...")
    ttft_params = SamplingParams(max_tokens=1, temperature=TEMPERATURE)
    t0 = time.perf_counter()
    llm.generate([TEST_PROMPT] * BATCH_SIZE, ttft_params)
    ttft = time.perf_counter() - t0
    print(f"  Time to first token (proxy): {ttft:.3f}s")

    print(f"\nThroughput benchmark ({N_BATCHES} batches x {BATCH_SIZE} prompts, {MAX_TOKENS} tokens each)...")
    print(f"  Prompt: {TEST_PROMPT[:60]}...")
    params = SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
    batch_results = []
    for batch_idx in range(N_BATCHES):
        t0 = time.perf_counter()
        outputs = llm.generate([TEST_PROMPT] * BATCH_SIZE, params)
        elapsed = time.perf_counter() - t0

        tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
        tok_per_sec = tokens_generated / elapsed

        print(f"\n  Batch {batch_idx + 1}/{N_BATCHES}")
        print(f"    Tokens generated : {tokens_generated}")
        print(f"    Total time       : {elapsed:.2f}s")
        print(f"    Speed            : {tok_per_sec:.1f} tok/s")

        batch_results.append({
            "batch": batch_idx + 1,
            "tokens_generated": tokens_generated,
            "elapsed_s": elapsed,
            "tok_per_sec": tok_per_sec,
        })
    print("=" * 60)

    # Persist the same statistics to disk (in addition to the stdout above) so runs
    # can be compared later without scraping the SGE .out logs.
    stats = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_repo": MODEL_REPO,
            "quantization": QUANTIZATION,
            "n_ctx": N_CTX,
            "max_tokens": MAX_TOKENS,
            "batch_size": BATCH_SIZE,
            "n_batches": N_BATCHES,
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "temperature": TEMPERATURE,
        },
        "gpu_memory_before": before,
        "gpu_memory_after": after,
        "ttft_proxy_s": ttft,
        "batches": batch_results,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = f"./benchmark_vllm_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
