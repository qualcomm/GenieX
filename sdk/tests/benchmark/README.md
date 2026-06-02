# geniex_benchmark — C inference benchmark example

Single-file C example that drives the public geniex C API. One invocation
runs one `(plugin, device, model)` cell (warmup + repeated measured runs)
and prints / writes TTFT, prefill_tps, decode_tps, gen_tokens.

The harness mirrors [`tests/benchmark/_runner.py`](../../../tests/benchmark/_runner.py)
exactly so C and Python report comparable numbers off the same SDK. It runs
from a **local model path** (a geniex bundle dir, or a `.gguf` file / its
folder) on Windows, Android, and Linux — the same binary feeds the scorecard.

## Build

Gated on the `GENIEX_BENCHMARK` cmake option, which the snapdragon presets in
[`sdk/CMakePresets.json`](../../CMakePresets.json) enable for both `debug` and
`release`. The recipes below match [`notes/build.md`](../../../notes/build.md).

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Shorten the source
> path with `subst` before building if the repo lives under a long path:
> `subst G: C:\path\to\geniex; cd G:\sdk`

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j --target geniex_benchmark
# → build\tests\benchmark\geniex_benchmark.exe
cmake --install build --prefix pkg-geniex   # optional → pkg-geniex\bin\geniex_benchmark.exe
```

### Linux (cross-compile from x86_64)

Build inside the Snapdragon Linux toolchain container per
[`notes/build.md`](../../../notes/build.md):

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  ghcr.io/qcom-ai-hub/geniex-toolchain-linux:v0.0.2 \
  bash -c 'cmake --preset arm64-linux-snapdragon-release -B build-linux . \
    && cmake --build build-linux -j --target geniex_benchmark \
    && cmake --install build-linux --prefix pkg-geniex'
# → pkg-geniex/bin/geniex_benchmark
```

### Android (cross-compile from Linux)

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  ghcr.io/qcom-ai-hub/geniex-toolchain-android:v0.0.1 \
  bash -c 'cmake --preset arm64-android-snapdragon-release -B build-android . \
    && cmake --build build-android -j --target geniex_benchmark \
    && cmake --install build-android --prefix pkg-geniex'
# → pkg-geniex/bin/geniex_benchmark
```

## Run

The binary loads `geniex.dll` / `libgeniex.so` and the per-plugin backends the
same way the Python binding does — from the installed `pkg-geniex/lib` (and
`lib/llama_cpp`) layout. On Windows run it from the build/install tree so the
DLLs resolve; on Android/Linux export `LD_LIBRARY_PATH=./lib:./lib/llama_cpp`
and `GENIEX_PLUGIN_PATH=./lib` (see [`notes/run.md`](../../../notes/run.md)).

```bash
# LLM, llama_cpp — point --model-path at a .gguf file
geniex_benchmark \
  --plugin llama_cpp --device hybrid \
  --model-path /path/to/Qwen3-0.6B-Q4_0.gguf

# LLM, QAIRT — the bundle dir is the "model path"
geniex_benchmark \
  --plugin qairt --device npu \
  --model-path /path/to/qualcomm/Qwen3-4B-Instruct-2507/

# VLM, llama_cpp — pass the model gguf + its mmproj (switches to VLM mode)
# and one or more --image. The prompt is run through the model's chat template
# (which places the image tokens) before generation.
geniex_benchmark \
  --plugin llama_cpp --device hybrid \
  --model-path /path/to/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj-path /path/to/mmproj-SmolVLM-500M-Instruct-f16.gguf \
  --image /path/to/sample.jpg

# VLM, QAIRT — the vision encoder is baked into the bundle (no mmproj), so
# pass --vlm to force VLM mode plus one or more --image
geniex_benchmark \
  --plugin qairt --device npu --vlm \
  --model-path /path/to/qualcomm/Qwen2.5-VL-7B-Instruct/ \
  --image /path/to/sample.jpg

# Customise: prompt, sample count, output files
geniex_benchmark \
  --plugin llama_cpp --device hybrid \
  --model-path .../Qwen3-1.7B-Q4_0.gguf \
  --warmup 1 --repeat 3 \
  --max-new-tokens 128 --temperature 0.0 --seed 42 \
  --output-json results/qwen3-1.7b-hybrid.json \
  --cell-id Qwen3-1.7B-llama_cpp-hybrid
```

On Windows the same invocations work with `.exe` and backslash paths, e.g.:

```powershell
build\tests\benchmark\geniex_benchmark.exe --plugin qairt --device npu `
  --model-path $env:USERPROFILE\.cache\geniex\models\qualcomm\Qwen3-4B-Instruct-2507
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
