# geniex_benchmark — C inference benchmark example

Single-file C example that drives the public geniex C API. One invocation
runs one `(plugin, device, model)` cell (warmup + repeated measured runs)
and prints / writes TTFT, prefill_tps, decode_tps, gen_tokens.

The harness mirrors [`tests/benchmark/_runner.py`](../../../tests/benchmark/_runner.py)
exactly so C and Python report comparable numbers off the same SDK.

## Build

Gated on the `GENIEX_BENCHMARK` cmake option. The standard presets in
`sdk/CMakePresets.json` enable it via the `debug` inheritance.

```bash
# Linux host (x86_64)
cmake --preset arm64-linux-snapdragon-debug -B build-linux .
cmake --build build-linux -j --target geniex_benchmark
cmake --install build-linux --prefix pkg-geniex
# → pkg-geniex/bin/geniex_benchmark
```

For Android, build inside the toolchain container per
[`notes/build.md`](../../../notes/build.md):

```bash
docker run --rm \
  --volume $(pwd):/workspace \
  --workdir /workspace/sdk \
  --platform linux/amd64 \
  ghcr.io/qcom-ai-hub/geniex-toolchain-android:v0.0.1 \
  bash -c 'cmake --preset arm64-android-snapdragon-debug -B build-android . \
    && cmake --build build-android -j --target geniex_benchmark \
    && cmake --install build-android --prefix pkg-geniex'
```

## Run

```bash
geniex_benchmark \
  --plugin llama_cpp \
  --device cpu \
  --model-path /path/to/Qwen3-0.6B-Q4_0.gguf

# QAIRT — the bundle dir is the "model path"
geniex_benchmark \
  --plugin qairt \
  --device npu \
  --model-path /path/to/qualcomm/Qwen3-4B-Instruct-2507/

# Customise: prompt, sample count, output files
geniex_benchmark \
  --plugin llama_cpp --device hybrid \
  --model-path .../Qwen3-1.7B-Q4_0.gguf \
  --warmup 1 --repeat 3 \
  --max-new-tokens 128 --temperature 0.0 --seed 42 \
  --output-json results/qwen3-1.7b-hybrid.json \
  --cell-id Qwen3-1.7B-llama_cpp-hybrid
```

Run `geniex_benchmark --help` for the full flag list.

## Defaults (match the Python harness)

- prompt: identical literal to [`tests/benchmark/_runner.py:28-31`](../../../tests/benchmark/_runner.py#L28-L31)
- `max_new_tokens=128`, `temperature=0.0`, `seed=42`
- `--warmup 1`, `--repeat 3` (median over 3 measured runs after 1 warmup)
- llama_cpp gets a `[warmup=i]` / `[run=i]` suffix appended to the prompt
  so the KV cache is busted between runs (mirrors `_runner.py:82-83`)

## Per-cell JSON shape

```json
{
  "schema_version": "1",
  "cell_id": "Qwen3-0.6B-llama_cpp-cpu",
  "plugin": "llama_cpp",
  "device": "cpu",
  "device_id": null,
  "model_path": ".../Qwen_Qwen3-0.6B-Q4_0.gguf",
  "params": { "warmup": 1, "repeat": 3, "max_new_tokens": 128, ... },
  "runs": [ { "run_idx": 0, "ttft_us": 49758, "prefill_tps": 102.1, ... }, ... ],
  "agg": {
    "ttft_ms":     {"median": 49.8, "min": 47.4, "max": 52.1},
    "prefill_tps": {"median": 102.1, ...},
    "decode_tps":  {"median": 60.9, ...},
    "gen_tokens":  {"median": 128},
    "prompt_tokens":{"median": 42}
  }
}
```

## Matrix-style runs

The matrix lives in [`tests/benchmark/_matrix.py`](../../../tests/benchmark/_matrix.py)
on the Python side. To get C-vs-Python comparable numbers, run the C
binary in **matrix mode** so a single `geniex_init` covers the whole
sweep — Hexagon FastRPC sessions and other plugin init costs are then
amortised across cells exactly as the Python harness does:

```bash
cat > matrix.tsv <<EOF
# cell_id<TAB>plugin<TAB>device<TAB>model_path[<TAB>tokenizer_path][<TAB>mmproj_path]
Qwen3-0.6B-llama_cpp-cpu	llama_cpp	cpu	/data/local/tmp/.cache/geniex/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-0.6B-llama_cpp-npu	llama_cpp	npu	/data/local/tmp/.cache/geniex/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-4B-qairt-npu	qairt	npu	/data/local/tmp/.cache/geniex/models/qualcomm/Qwen3-4B-Instruct-2507
EOF

geniex_benchmark --matrix-file matrix.tsv --output-json-dir results/
```

For a one-cell-per-process invocation (cold-start each time, useful as
the reference for a customer-facing single-call workload), pass
`--plugin / --device / --model-path` directly without `--matrix-file`.

Models must be pre-pulled (e.g. `geniex-py pull bartowski/Qwen_Qwen3-0.6B-GGUF:Q4_0`)
before invoking this binary — the C side does not include a model
puller.

## Why no doctest? Why no model manager dependency?

The earlier `sdk/tests/` C++ doctest tree was unused in CI and overlapped
the Python e2e suite. It is replaced by this single, dependency-free C
example. Caching, alias resolution, and matrix orchestration stay on the
Python side ([`bindings/python/geniex/cli.py`](../../../bindings/python/geniex/cli.py),
[`tests/benchmark/run.py`](../../../tests/benchmark/run.py)); the C
binary stays small and exercises only the public API.
