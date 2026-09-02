# geniex-bench — C inference benchmark example

C example that drives the public geniex C API. One invocation runs one
`(plugin, device, model)` cell (warmup + repeated measured runs) and prints /
writes TTFT, prefill_tps, decode_tps, gen_tokens.

The sources are split by concern — `benchmark.c` holds `main()` and the
single-cell / matrix drivers, with argv parsing, model resolution, the run
loops, aggregation and report writing in their own files. See
[`bench.h`](bench.h) for the module map and the shared types.

Flag naming follows llama.cpp's
[`llama-bench`](../../third-party/llama.cpp/tools/llama-bench/README.md) —
`-r / --repetitions`, `-n / --n-gen`, `-c / --ctx-size`, `-t / --threads`,
`-m / --model`, `--n-gpu-layers`, `--no-warmup` — so users moving between
the two tools read the same vocabulary.

It accepts either a **local model path** (a geniex bundle dir, or a `.gguf`
file / its folder) or a **model-manager id** (`org/repo[:quant]`); ids are
resolved via the `geniex_model_*` C API — downloading on first use and
reusing the cached copy thereafter. Runs on Windows, Android, and Linux —
the same binary feeds Geniex Bench.

## Build

Gated on the `GENIEX_BENCHMARK` cmake option, which the snapdragon presets in
[`sdk/CMakePresets.json`](../CMakePresets.json) enable for both `debug` and
`release`. The recipes below match [`notes/build.md`](../../notes/build.md).

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Shorten the source
> path with `subst` before building if the repo lives under a long path:
> `subst G: C:\path\to\geniex; cd G:\sdk`

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j --target geniex-bench
# → build\benchmark\geniex-bench.exe
cmake --install build --prefix pkg-geniex   # optional → pkg-geniex\bin\geniex-bench.exe
```

### Linux (cross-compile from x86_64)

Build inside the Snapdragon Linux toolchain container per
[`notes/build.md`](../../notes/build.md):

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  docker.io/qualcomm/geniex-toolchain-linux:v0.1.0 \
  bash -c 'cmake --preset arm64-linux-snapdragon-release -B build-linux . \
    && cmake --build build-linux -j --target geniex-bench \
    && cmake --install build-linux --prefix pkg-geniex'
# → pkg-geniex/bin/geniex-bench
```

### Android (cross-compile from Linux)

```bash
docker run --rm -u $(id -u):$(id -g) \
  --volume $(pwd):/workspace --workdir /workspace/sdk \
  --platform linux/amd64 \
  docker.io/qualcomm/geniex-toolchain-android:v0.1.0 \
  bash -c 'cmake --preset arm64-android-snapdragon-release -B build-android . \
    && cmake --build build-android -j --target geniex-bench \
    && cmake --install build-android --prefix pkg-geniex'
# → pkg-geniex/bin/geniex-bench
```

## Run

The binary loads `geniex.dll` / `libgeniex.so` and the per-plugin backends the
same way the Python binding does — from the installed `pkg-geniex/lib` (and
`lib/llama_cpp`) layout. On Windows run it from the build/install tree so the
DLLs resolve; on Android/Linux export `LD_LIBRARY_PATH=./lib:./lib/llama_cpp`
and `GENIEX_PLUGIN_PATH=./lib` (see [`notes/run.md`](../../notes/run.md)).

```bash
# LLM, llama_cpp — point -m at a .gguf file
geniex-bench \
  --plugin llama_cpp --device hybrid \
  -m /path/to/Qwen3-0.6B-Q4_0.gguf

# LLM, QAIRT — the bundle dir is the "model path"
geniex-bench \
  --plugin qairt --device npu \
  -m /path/to/qualcomm/Qwen3-4B-Instruct-2507/

# VLM, llama_cpp — pass the model gguf + its mmproj (switches to VLM mode)
# and one or more --image. The prompt is run through the model's chat template
# (which places the image tokens) before generation.
geniex-bench \
  --plugin llama_cpp --device hybrid \
  -m /path/to/SmolVLM-500M-Instruct-Q8_0.gguf \
  --mmproj-path /path/to/mmproj-SmolVLM-500M-Instruct-f16.gguf \
  --image /path/to/sample.jpg

# VLM, QAIRT — the vision encoder is baked into the bundle (no mmproj), so
# pass --vlm to force VLM mode plus one or more --image
geniex-bench \
  --plugin qairt --device npu --vlm \
  -m /path/to/qualcomm/Qwen2.5-VL-7B-Instruct/ \
  --image /path/to/sample.jpg

# GPU (llama_cpp) — the gpu alias selects GPUOpenCL and offloads all layers by
# default (-1); pass --n-gpu-layers to offload only some
geniex-bench \
  --plugin llama_cpp --device gpu \
  -m /path/to/Qwen3-4B-Q4_K_M.gguf

# Customise: prompt, sample count, output files
geniex-bench \
  --plugin llama_cpp --device hybrid \
  -m .../Qwen3-1.7B-Q4_0.gguf \
  --warmup 1 -r 3 \
  -n 128 --temperature 0.0 --seed 42 \
  --output-json results/qwen3-1.7b-hybrid.json \
  --cell-id Qwen3-1.7B-llama_cpp-hybrid

# Accuracy mode: single run, print the generated text (eyeball output quality,
# not speed). Pair with --prompt-file so the model sees a real prompt.
geniex-bench \
  --plugin llama_cpp --device hybrid \
  -m .../Qwen3-1.7B-Q4_0.gguf \
  --accuracy --prompt-file prompt.txt -n 128

# Logits mode: one prefill-only forward pass (no decode loop), write every
# position's top-N logits to JSON for on-target accuracy metrics (perplexity,
# MMLU, MMMU). Input is random ids only (-p N); add --logits-last-only for just
# the last token's row.
geniex-bench \
  --plugin llama_cpp --device npu \
  -m .../Qwen3-1.7B-Q4_0.gguf \
  --logits -p 128 --logits-top-n 20 --output-json logits.json
```

On Windows the same invocations work with `.exe` and backslash paths, e.g.:

```powershell
build\benchmark\geniex-bench.exe --plugin qairt --device npu `
  -m $env:USERPROFILE\.cache\geniex\models\qualcomm\Qwen3-4B-Instruct-2507
```

Run `geniex-bench --help` for the full flag list.

## Defaults

- `n_gen=128`, `temperature=0.0`, `seed=42`
- `--warmup 1`, `-r 5` (5 measured runs after 1 warmup; pass `--no-warmup`
  to skip warmup)
- `--accuracy` pins a single run (`--warmup 0 -r 1`) and prints the generated
  text to stdout (`[gen ] ...`); use it to sanity-check output quality rather
  than timing. Pair with `--prompt-file`, since the default random-ids prefill
  yields meaningless text. Only in this mode does a line that is exactly `---`
  split the prompt file into several prompts: each runs on its own KV cache, is
  marked `[sep ] prompt i/n` on stdout, and gets its own entry in the JSON
  `runs` array. Every other mode feeds the file verbatim as one prompt,
  because the segments differ in length and a tok/s median across them would
  mix populations.
- with `--accuracy`, each prompt-file segment is run through the bundle's own
  chat template (`geniex_llm_apply_chat_template`) before generation — the
  same templating `geniex infer` uses — so pass the raw user turn, not
  pre-templated text. `--system-prompt TEXT` adds a system message ahead of
  it; `--think` / `--no-think` sets `enable_thinking` (default: think).
  Without `--accuracy`, `--prompt-file` still feeds the file verbatim.
- `--logits` runs one prefill-only forward pass (no decode loop, no timing) over
  `-p N` random ids and writes every position's top-N logits to `--output-json`
  (`--logits-top-n`, default 20; `--logits-last-only` for the last row only).
  Both llama_cpp and qairt support it; `--prompt-file` is rejected since the
  forward-logits API takes `input_ids` and the tool has no tokenizer.
- llama_cpp gets a `[warmup=i]` / `[run=i]` suffix appended to the prompt
  so the KV cache is busted between runs
- for `--plugin qairt`, `prompt_tokens` and `prefill_tps` are reported over the
  padded prompt length `ceil(prompt_tokens / 128) * 128`: the QAIRT engine pads
  input_ids to a 128-token prefill chunk, so the padded count reflects the work
  actually done (#1194). llama_cpp does no such padding and is reported as-is

## Per-cell JSON shape

```json
{
  "schema_version": "2",
  "cell_id": "Qwen3-0.6B-llama_cpp-cpu",
  "plugin": "llama_cpp",
  "device": "cpu",
  "device_id": null,
  "model_path": ".../Qwen_Qwen3-0.6B-Q4_0.gguf",
  "model_size_bytes": 368705536,
  "params": { "warmup": 1, "repetitions": 3, "n_gen": 128, ... },
  "runs": [ { "run_idx": 0, "ttft_us": 49758, "prefill_tps": 102.1, ... }, ... ],
  "agg": {
    "ttft_ms":     {"median": 49.8, "min": 47.4, "max": 52.1, "mean": 49.7, "stdev": 2.4},
    "prefill_tps": {"median": 102.1, "min": 98.0, "max": 110.3, "mean": 103.4, "stdev": 6.2},
    "decode_tps":  {"median": 60.9, "min": 58.1, "max": 62.5, "mean": 60.5, "stdev": 2.3},
    "gen_tokens":  {"median": 128},
    "prompt_tokens":{"median": 42}
  }
}
```

## Accuracy mode output

`--accuracy` prints the generated text to stdout, one `[gen ]`-prefixed line
per output line, then the usual `[ok  ]` summary line:

```
[gen ] The capital of France is Paris.
[gen ] Answer: Paris
[ok  ] cell  plugin=llama_cpp device=cpu ngl=0 ttft=475.6ms prefill=25.3tps decode=18.2tps gen=24 tok
```

## Logits mode JSON shape

`--logits` writes its own report (`schema_version` `logits-1`): shape metadata
plus `rows`, one row per emitted position, each a top-N array of
`[token_id, logit]` pairs sorted by descending logit.

```json
{
  "schema_version": "logits-1",
  "cell_id": "cell",
  "plugin": "llama_cpp",
  "device": "npu",
  "model_path": ".../Qwen3-1.7B-Q4_0.gguf",
  "n_gpu_layers": 999,
  "n_prompt": 128,
  "all_positions": true,
  "n_rows": 128,
  "vocab_size": 151936,
  "top_n": 20,
  "truncated_to_top_n": true,
  "rows": [
    [[9, 6.950917], [1479, 6.472050], ...],
    ...
  ]
}
```

## Markdown row shape

`--output-md` (and the QDC bench report) produce a llama-bench-aligned table:

```
| Model     | Size    | Backend    | Device | ngl | Test       | TTFT (ms)   | Prefill (tok/s) | Decode (tok/s) |
|-----------|--------:|------------|--------|----:|------------|------------:|----------------:|---------------:|
| Qwen3-0.6B| 351 MiB | llama_cpp  | cpu    |   - | pp42+tg128 | 49.8 ± 2.4  |  102.1 ± 6.2    |  60.9 ± 2.3    |
```

## Matrix-style runs

Run the C binary in **matrix mode** so a single `geniex_init` covers the
whole sweep — Hexagon FastRPC sessions and other plugin init costs are
then amortised across cells:

```bash
cat > matrix.tsv <<EOF
# cell_id<TAB>plugin<TAB>device<TAB>model_path[<TAB>tokenizer_path][<TAB>mmproj_path]
Qwen3-0.6B-llama_cpp-cpu	llama_cpp	cpu	/data/local/tmp/.cache/geniex/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-0.6B-llama_cpp-npu	llama_cpp	npu	/data/local/tmp/.cache/geniex/models/bartowski/Qwen_Qwen3-0.6B-GGUF/Qwen_Qwen3-0.6B-Q4_0.gguf
Qwen3-4B-qairt-npu	qairt	npu	/data/local/tmp/.cache/geniex/models/qualcomm/Qwen3-4B-Instruct-2507
EOF

geniex-bench --matrix-file matrix.tsv --output-json-dir results/
```

For a one-cell-per-process invocation (cold-start each time, useful as
the reference for a customer-facing single-call workload), pass
`--plugin / --device / -m` directly without `--matrix-file`.

Either a path or a model-manager id works in column 4; the binary calls
`geniex_model_pull` on first use and reuses the cached copy on subsequent
cells. Pre-pulling with `geniex-py pull ...` is still supported and skips
the cold download.

```bash
cat > matrix.tsv <<EOF
# cell_id<TAB>plugin<TAB>device<TAB>model_path_or_id
Qwen3-0.6B-cpu	llama_cpp	cpu	bartowski/Qwen_Qwen3-0.6B-GGUF:Q4_0
Qwen3-4B-qairt	qairt	npu	qualcomm/qwen3_4b
EOF
geniex-bench --matrix-file matrix.tsv --output-json-dir results/ \
  --mm-data-dir ./cache --chipset qualcomm-snapdragon-x-elite
```

## Why a model-manager dependency now?

The earlier `sdk/tests/` C++ doctest tree was unused in CI and overlapped
the Python e2e suite. It was replaced by this single C example. Caching,
alias resolution, and matrix orchestration originally stayed on the
Python side; the QDC bench run ran `curl` / `Invoke-WebRequest` on each
device for every model. That serial download was the slowest phase of
the bench run and OOMed on large GGUFs on Windows. Linking the C binary
against `libgeniex_model` and resolving column-4 model ids via
`geniex_model_pull` collapses the device-side download to one
multi-connection, resumable HTTPS call — and exercises the same model
manager our Python / Go / JNI bindings ship to users.
