# Run

Terminology used throughout this doc:

- **Runtime** — `llama_cpp` or `qairt` (the `plugin_id`).
- **Compute unit** — NPU, GPU, CPU, or `hybrid` (mapped from `--device` / `device_id`).
- **Chipset** — the SoC, e.g. Snapdragon X Elite (`SM8750`, `SM8850`, …).

Two runtimes ship with geniex and both can drive the Snapdragon NPU, but through **separate user-space stacks** that consume **different model formats**:

- **`llama_cpp`** — GGUF models, targets Hexagon NPU (via `ggml-hexagon`), Adreno GPU (via OpenCL), or CPU.
- **`qairt`** — QAIRT `.bin` shards, targets Hexagon NPU via Qualcomm's QNN runtime.

They are not interchangeable; the runtime is chosen per model.

> For the CI/S3 signing pipeline that backs HTP releases, see [release.md § Hexagon HTP signing](release.md#hexagon-htp-signing).

## Compute-unit aliases

The alias table lives in the **SDK**, not in the bindings:
[`sdk/src/device.cpp`](../sdk/src/device.cpp) exposes
`geniex_resolve_device` via `sdk/include/geniex.h`. The Go wrapper
([`bindings/go/device.go`](../bindings/go/device.go)), Python wrapper
(`resolve_device` in
[`bindings/python/geniex/_ffi/_api.py`](../bindings/python/geniex/_ffi/_api.py)),
and Android/JNI wrapper (`resolve_device` in
[`bindings/android/app/src/main/cpp/jniutils.cpp`](../bindings/android/app/src/main/cpp/jniutils.cpp))
are all thin FFI shims over that one function. Editing alias semantics
means editing `sdk/src/device.cpp`, rebuilding the SDK bridge
(`/build`), and possibly updating all three FFI stubs if the struct
shape changes (see [CONTRIBUTING.md](../CONTRIBUTING.md) for the
FFI-sync rule).

| Alias    | `device_id` sent to SDK | `n_gpu_layers` override | Use case                                                                                    |
|----------|-------------------------|-------------------------|---------------------------------------------------------------------------------------------|
| `cpu`    | empty                   | `0`                  | Pure CPU.                                                                                   |
| `gpu`    | `GPUOpenCL`             | `--ngl` (default -1) | Adreno via OpenCL.                                                                          |
| `npu`    | `HTP0`                  | `--ngl` (default -1) | Pinned single-session HTP. Deterministic, slower on LLMs — see § NPU compute-unit selection (llama_cpp). |
| `hybrid` | empty                   | `--ngl` (default -1) | `llama_cpp` per-tensor HTP+CPU scheduler.                                                    |

`--ngl` defaults to `-1`, which llama.cpp reads as "all layers", so
gpu / npu / hybrid offload everything unless `--ngl` is set. The value
passes through the SDK unchanged. qairt ignores `--ngl` (forced to 0).

Defaults when the user passes nothing (`--device ""` / `device_map="auto"`):
`npu` for both `llama_cpp` and `qairt`. QAIRT exposes only one device,
so `cpu` / `gpu` / `hybrid` against a qairt model get coerced to `NPU`
with a warning on stderr — the CLI does **not** exit early.

Beyond the aliases, `--compute` also accepts an explicit device list of
concrete ids (`HTP0,HTP1,HTP2,HTP3`, `GPUOpenCL`) — `llama_cpp` passes it
through to llama.cpp verbatim (handy for multi-DSP recipes that need more
than the single `HTP0` the `npu` alias pins); `--ngl` still applies. qairt
is NPU-only, so a device list gets coerced to `NPU` with a warning.

## Power mode

Like compute-unit aliases, the power-mode alias table lives in the **SDK**:
[`sdk/src/device.cpp`](../sdk/src/device.cpp) exposes
`geniex_resolve_power_mode` via `sdk/include/geniex.h`. It's a single
unified knob for both runtimes' HTP DCVS/HMX power/clock-management, set via
`geniex_ModelConfig.power_mode` (`--power-mode` on `geniex infer` / `run` /
`serve` and `geniex-bench`; a JSON `power_mode` field on `geniex serve`
requests).

| Alias                          | Relative power (lowest → highest) |
|---------------------------------|-----|
| `low_power_saver`               | 1   |
| `power_saver`                   | 2   |
| `high_power_saver`              | 3   |
| `low_balanced`                  | 4   |
| `balanced`                      | 5   |
| `high_performance`              | 6   |
| `sustained_high_performance`    | 7   |
| `burst`                         | 8 (highest — max clocks, DCVS disabled) |

Empty / `default` resolves to `burst` on both runtimes, matching today's
behavior before this knob existed (qairt's `perf_profile` already defaulted
to `BURST`; llama_cpp's HTP corners were hardcoded to max). This applies
only on the NPU compute unit — it's a no-op (logged, not an error) on
`cpu` / `gpu`.

- **`llama_cpp`** — carried by an internal, unmerged patch on top of
  [`ggml/llama.cpp#340`](https://github.qualcomm.com/ggml/llama.cpp/pull/340)
  (`sdk/patches/llama-hexagon-power-mode.patch` +
  `sdk/patches/llama-hexagon-power-mode-setter.patch`; see the comments in
  [`sdk/CMakeLists.txt`](../sdk/CMakeLists.txt)). Only affects sessions
  created after the call — an already-open HTP session (another model still
  loaded in the same process) keeps its old mode until released and
  reacquired; the plugin logs a warning rather than silently no-op'ing.
- **`qairt`** — sets `ModelConfig::perf_profile`, which the QAIRT core (a
  real HTP `PerfProfile` → `QnnHtpPerfInfrastructure` vote) already
  supports. **Bundle precedence:** if the model bundle ships an
  `htp_backend_ext_config.json` with its own `perf_profile`, that value
  wins over `--power-mode` (`Model::initialize` seeds from the caller, then
  lets the bundle's `parseHtpConfig` overwrite it) — the plugin logs a
  warning when the two disagree.

## Compute-unit selection (llama_cpp)

`llama_cpp` supports OpenCL and Hexagon on Windows ARM64. The compute unit is driven by two inputs on `geniex_LlmCreateInput`:

- `device_id` — string, runtime-specific (`HTP0`, `GPUOpenCL`, `CPU`, …).
- `config.n_gpu_layers` — int; how many layers to offload. `-1` = all.

### NPU compute-unit selection (llama_cpp)

`sdk/plugins/llama_cpp/src/llm.cpp:73-114` branches on whether `device_id` is non-null, producing **two runtime paths** with very different performance:

1. **`device_id` null + `n_gpu_layers=-1`** (the `hybrid` alias) → llama.cpp's **per-tensor scheduler**. It inspects each tensor and assigns it to whichever registered backend supports the op (HTP for computable ops, CPU for fallbacks), using CPU-resident buffers for the fallback tensors. **Fast path.** On X1E80100 + Qwen3-1.7B-Q8_0: ~90 tok/s prefill, ~27 tok/s decode, ~200 ms TTFT. Task Manager shows NPU pegged.

2. **`device_id="HTP0"` + `n_gpu_layers=-1`** (the `npu` alias) → runtime calls `ggml_backend_dev_by_name("HTP0")` and sets `mpar.devices = {HTP0}`. This **pins the model to a single compute-unit layout** and disables per-tensor hybrid assignment. Any op HTP doesn't support gets handled less efficiently. On the same model: ~60 tok/s prefill, ~22 tok/s decode, ~350 ms TTFT. Task Manager shows CPU pegged (the host thread driving HTP busy-waits, *plus* all fallbacks run there). Useful when you want deterministic layout / all weights on a known compute unit. Note: a non-zero `n_gpu_layers` is required even with the compute unit pinned — `device_id="HTP0"` with `ngl=0` opens an HTP session and then runs every layer on CPU, so the default `ngl=-1` (all layers) applies for this alias (`sdk/src/device.cpp`).

Bonus: when the `device_id` string starts with `"HTP0"`, the runtime also flips KV cache to Q8_0 and enables flash-attn (`llm.cpp:136-140`). Orthogonal to perf — path (2) is slower than (1) even with those enabled.

**Rule of thumb:** use `--device hybrid` (or leave `--device` empty) for fastest throughput; use `--device npu` when you need determinism or when debugging placement.

History: the `fb98467` commit ("add device parameter") originally made `--device npu` synthesize `device_id="HTP0"`, collapsing the fast path. That was reverted (hybrid became the implicit default), then the two semantics were split into explicit `npu` / `hybrid` aliases to let callers pick.

### Running from the CLI

Use `--device` (`-d`):

```powershell
geniex infer Qwen/Qwen3-1.7B-GGUF                 # hybrid (default) for llama.cpp
geniex infer Qwen/Qwen3-1.7B-GGUF --device npu    # pinned HTP0
geniex infer Qwen/Qwen3-1.7B-GGUF --device hybrid # explicit hybrid
geniex infer Qwen/Qwen3-1.7B-GGUF --device gpu
geniex infer Qwen/Qwen3-1.7B-GGUF --device cpu
```

### Sanity-checking which path actually ran

The SDK's default log handler is a no-op in release builds (`sdk/src/ml.cpp:36-60`), so `stdout`/`stderr` stays silent and "did it actually use HTP?" is easy to guess wrong. Ways to check:

- **Python:** set `GENIEX_LOG=INFO`. The Python binding installs a `geniex_set_log` callback that routes SDK messages (`Found device: HTP0`, `Using N device(s)`, etc.) to stderr. If you see `Found device: …` lines you're on the **pinned-`HTP0` path** (the `npu` alias); absence = hybrid path.
- **Windows:** Task Manager's NPU graph. Hybrid lights it up; pinned-`HTP0` pegs the CPU (host thread busy-waits HTP the whole inference).
- **Signature:** on Snapdragon X1E80100 + a 1.7B Q8_0 model, hybrid gives prefill ≳ 80 tok/s and TTFT ≲ 250 ms; pinned-`HTP0` gives prefill ≲ 65 tok/s and TTFT ≳ 340 ms. Prefill and TTFT separate the two paths more cleanly than decode.
- **Threadpool:** any offloaded path (all aliases except `cpu`) logs `threadpool tuned for offload: 6 threads pinned to cores [2, 8), strict, poll=1000` — the SDK mirrors upstream's fixed `-t 6 --cpu-mask 0xfc` (`sdk/plugins/llama_cpp/src/threadpool.cpp`); pass `n_threads` in `geniex_ModelConfig` to override. Absent on `--device cpu`.

If you see `Device '…' not found, skipping`, the runtime loaded but the GGML backend DLL did not — verify test-signing is still on (for HTP) or that `ggml-opencl.dll` is present in `sdk/pkg-geniex/lib/llama_cpp/`.

> Q4_K_M is a suboptimal quant for HTP — it prefers Q4_0 / Q8_0, so some tensors fall back to CPU. Use Q4_0 for a clean NPU run.

## Running QAIRT models

QAIRT exposes only its Hexagon NPU compute unit (`plugin_id="qairt"`, `device_id="NPU"`). The SDK's `geniex_resolve_device` coerces `--device cpu` / `gpu` / `hybrid` to `npu` with a stderr warning so existing shell pipelines don't break — expect a line like:

```
Warning: qairt plugin only supports NPU inference; ignoring device='cpu' and running on NPU
```

QAIRT models need a `geniex.json` to work. See the [granite4_micro example](https://huggingface.co/yichqian/geniex-qairt-models/blob/main/granite4_micro/geniex.json).

### Using a custom QNN library

A QAIRT runtime ships with GenieX and is used by default. To run against a different one
without reinstalling, point the plugin at it:

```bash
# via the CLI flag (qairt models only)
geniex infer local/granite4_micro --qairt-lib /path/to/qairt/2.XX.0

# or via the environment variable (picked up by any front-end: CLI, pybind, Android)
GENIEX_QAIRT_LIB=/path/to/qairt/2.XX.0 geniex infer local/granite4_micro
```

SDK embedders call it **before `geniex_init`** instead of touching their own environment —
the only route on Android, where the JVM cannot `setenv`:

```c
geniex_set_qairt_runtime_path("/path/to/qairt/2.XX.0");  /* "" restores the bundled runtime */
geniex_init();
```

```python
geniex.set_qairt_runtime_path('/path/to/qairt/2.XX.0')
```

```kotlin
GenieXSdk.getInstance().setQairtRuntimePath("/path/to/qairt/2.XX.0")
```

Precedence is `geniex_set_qairt_runtime_path` → `GENIEX_QAIRT_LIB` → the bundled runtime.
`--qairt-lib` calls the API, so the flag wins over an inherited environment variable.

The path accepts either layout:

- **A QAIRT SDK root** (as installed from the Qualcomm Software Center). The host libraries
  are taken from `lib/<triple>` (`aarch64-windows-msvc`, `aarch64-android`, or
  `aarch64-oe-linux-gcc11.2`) and `ADSP_LIBRARY_PATH` is pointed at every Hexagon DSP skel
  folder (`lib/hexagon-v*/unsigned`), so the on-device HTP arch is matched automatically.
- **A flat folder** holding `QnnHtp.dll` and `QnnSystem.dll` (or the `libQnn*.so`
  equivalents) directly — the same shape as the bundled `htp-files` layout.

> [!IMPORTANT]
> A runtime the plugin accepts can still produce **wrong output at full speed**, so confirm
> the override resolved where you meant. Run with `--log info` (the CLI default is `none`):
>
> ```
> Overriding the bundled QAIRT runtime from <source>: <what you passed> (host libs: <resolved dir>)
> ```
>
> `host libs:` is the part that matters — for an SDK root it is the `lib/<triple>` subfolder,
> not the root you passed. `<source>` is `geniex_set_qairt_runtime_path` or
> `GENIEX_QAIRT_LIB`, so the line also tells you which knob won.

<details><summary>Which runtimes are accepted, and why this is a supported override</summary>

The plugin reaches QNN only through the versioned C interface, which negotiates at load time
against `compiled QNN_API_VERSION_MINOR <= runtime minor`. The floor is the API version the
plugin *compiled* against (QAIRT 2.36 headers, C API 2.27), not the version it bundles — so
runtimes both older and newer than the bundled one are accepted, down to that floor. One
build was measured on Snapdragon X Elite across 2.45 / 2.48 / 2.49 at identical throughput.

This replaces an earlier C++ `IBackend` path whose vtable layout changed every release, where
a mismatch could segfault (ai-hub-models-internal#3964). That hazard is gone with the C API,
which is why this is a supported override rather than a testing-only aid.

`QnnHtpNetRunExtensions` is no longer loaded at all — the plugin applies
`htp_backend_ext_config.json` itself through the public C API — so a runtime folder does not
need to carry it.

An unusable path fails model load immediately, naming the source it came from, e.g.
`geniex_set_qairt_runtime_path does not contain QnnHtp.dll (looked in the folder itself and
lib/aarch64-windows-msvc): <path>`.

**One runtime per process.** `QnnHtp` is loaded once and stays resident for the life of the
process — the plugin never unloads it, and `ADSP_LIBRARY_PATH` / `SetDllDirectory` are
process-wide. So the path is locked at `geniex_init`, and setting it afterwards returns
`GENIEX_ERROR_COMMON_ALREADY_INITIALIZED` rather than appearing to work. `geniex_deinit` does
not unlock it: the QNN libraries outlive the cycle. Comparing versions means one process per
version.

</details>

### HTP multicore (qairt)

By default a QAIRT model executes on **one** NSP core. The knob is the bundle's
`htp_backend_ext_config.json`: the QAIRT core counts the `devices[].cores` entries
and, when more than one is listed, sets
`QNN_HTP_GRAPH_CONFIG_OPTION_NUM_CORES` on every loaded graph
(`ModelConfig::num_cores` in the plugin; see
[geniex-qairt docs § HTP Backend Config and Multicore Execution](../third-party/geniex-qairt/docs/README.md#5-htp-backend-config-and-multicore-execution)).

What you can and cannot change at load time:

- **Load time**: the `NUM_CORES` graph config, per-core `perf_profile`, and
  `rpc_control_latency` — all read from the JSON when the model loads.
- **Generation time**: graph partitioning baked into the context binaries.
  Binaries compiled single-core may not speed up (or may reject the config)
  even when more cores are requested.

At init the runtime logs the device-reported core count and the effective core
count (`--log info`), warns and clamps when the JSON requests more cores
than the SoC exposes, and warns without failing when the driver reports
multicore unavailable (QNN verbose trace: `Error code 1000 ... key = 304` /
`Multicore support is unavailable`). Most current Snapdragon mobile/compute
SoCs expose a single NSP core; multi-NSP parts (e.g. certain automotive SoCs)
report `numCores > 1` in QNN platform info.

### Build and run locally

```bash
hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models
bazelisk run //cli -- pull local/granite4_micro \
  --model-hub localfs \
  --local-path /absolute/path/to/geniex-qairt-models/granite4_micro
bazelisk run //cli -- infer local/granite4_micro
```

### Hand off a build to another machine

Builder:

```bash
bazelisk build //cli:artifact
# Export bazel-bin/cli/artifact.zip and ggml-htp-v1.cer
```

Recipient:

```bash
# unzip artifact.zip
hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models
./geniex.exe pull local/granite4_micro --model-hub localfs \
  --local-path /absolute/path/to/geniex-qairt-models/granite4_micro
./geniex.exe infer local/granite4_micro
```

## Running a prebuilt CI release (Windows on Snapdragon)

Every `v*` tag publishes the Windows ARM64 installer on [the Releases page](https://github.com/qualcomm/GenieX/releases). Download both:

- `geniex-cli-setup.exe` — the installer
- `geniex-sdk-windows-arm64-<tag>.zip` — the SDK

The SDK filename encodes the HTP signing flavor:

| Filename                                        | HTP signing        | Extra setup                                 |
|-------------------------------------------------|--------------------|---------------------------------------------|
| `geniex-sdk-windows-arm64-<tag>.zip`            | Microsoft-signed   | None — skip to "Run" below.                 |
| `geniex-sdk-windows-arm64-<tag>-selfsigned.zip` | Self-signed (test) | See [Self-signed fallback](#self-signed-fallback). |

If the release also attaches `ggml-htp-v1.cer`, you're on the self-signed flavor.

Run:

1. Install with `geniex-cli-setup.exe`.
2. `hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models`
3. `geniex.exe pull local/granite4_micro --model-hub localfs --local-path <abs-path>\geniex-qairt-models\granite4_micro`
4. `geniex.exe infer local/granite4_micro`

## Self-signed fallback

Only needed when the release ships the `-selfsigned` SDK plus `ggml-htp-v1.cer`. Windows refuses to load `libggml-htp.cat` until you both enable test signing **and** trust the cert.

**Pre-built users** already have `ggml-htp-v1.cer` from the release page — skip ahead to step 2 below.

**Builders** (generating their own cert for a local build): run these in elevated `cmd.exe`. The `.pfx` is what `HEXAGON_HTP_CERT` needs at build time; the `.cer` is what's imported into the trust stores.

```cmd
set "PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\arm64;%PATH%"
mkdir C:\Users\%USERNAME%\Certs
cd C:\Users\%USERNAME%\Certs
makecert -r -pe -ss PrivateCertStore -n CN=GGML.HTP.v1 -eku 1.3.6.1.5.5.7.3.3 -sv ggml-htp-v1.pvk ggml-htp-v1.cer
pvk2pfx -pvk ggml-htp-v1.pvk -spc ggml-htp-v1.cer -pfx ggml-htp-v1.pfx
setx /M HEXAGON_HTP_CERT "C:\Users\%USERNAME%\Certs\ggml-htp-v1.pfx"
```

`makecert` prompts twice for a password — leave blank for a throwaway dev cert. Do **not** reuse a `.cer` extracted from someone else's signed binary: it has no private key (so it's unusable for signing) and importing a random third-party root is a security risk.

Then, for both builders and pre-built users:

1. **Enable test signing** (elevated PowerShell, then reboot):

   ```powershell
   bcdedit /set TESTSIGNING ON
   ```

   If this fails with a Secure Boot error, disable Secure Boot in UEFI first, then retry.

2. **Import `ggml-htp-v1.cer` into two stores** via `certlm.msc` (must be launched elevated, else imports fail with "store was read only"):

   - `Trusted Root Certification Authorities` → `Certificates` → right-click → **All Tasks → Import…** → select `ggml-htp-v1.cer`.
   - Repeat into `Trusted Publishers` → `Certificates`.

   Both stores are required: Root makes the chain valid; Trusted Publishers suppresses the driver-load prompt.

3. Reboot if you haven't yet. Verify:

   ```powershell
   bcdedit /enum | Select-String testsigning   # should show "testsigning   Yes"
   ```

Upstream background: `third-party/llama.cpp/docs/backend/snapdragon/windows.md`.

## Update checks

geniex consults a cached "latest release" entry and prints a one-line notice if a newer version exists, at most once per 8 h. A background refresh re-fetches at most every 24 h.

The version data comes from a public S3 index (`qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/index.json`). Failures are silent (logged at debug only, no stdout spam).

Run `geniex update` to upgrade:

- **Windows** — downloads and launches the signed installer once it's published; otherwise reports "up-to-date".
- **Linux** — prints the install-script one-liner to re-run (`curl -fsSL … | bash`); auto-update is not wired up yet.

Pass `--skip-update` on any command to skip the probe (and the notify banner) entirely for that invocation.

## Performance metrics

`--verbose` (and the Go/Python APIs' `ProfileData`, and `geniex-bench`) report one number per inference phase. Which metric covers which phase:

| Metric | Field | Phase it measures |
|--------|-------|-------------------|
| `ttft` | `ttft` | Start of generate → first sampled token. For a VLM this **includes** the media encoder, so it is *not* comparable to a pure prefill number. |
| media time | `media_time` | The vision/audio **encoder** only — turning pixels/audio into decoder-space embeddings. `0` on text-only runs. |
| prompt / prefill time | `prompt_time` | Prefill — running the prompt tokens through the model. On a VLM this includes prefilling the media (soft) tokens; it excludes the encoder. |
| prompt / prefill speed | `prefill_speed` | `prompt_tokens / prompt_time`. |
| decode time / speed | `decode_time` / `decoding_speed` | Generation phase (first token → last token). |

Key points:

- **`media_time` is the encoder only.** Everything downstream of the encoder (prefilling the media tokens through the model) lives in `prompt_time`, same as text.
- **`prompt_tokens` counts text + media tokens** on a VLM run, so `prefill_speed` reflects the full prefill the model actually did.
- `ttft` spans encoder + prefill, so `ttft ≈ media_time + prompt_time` for a VLM.

Both runtimes measure `media_time` at the same boundary (encoder wall time only), so the numbers are comparable across plugins:

- **llama.cpp** — timed per-chunk: the encode (`mtmd_encode_chunk`) is `media_time`; prefilling the embeddings (`mtmd_helper_decode_image_chunk` → `llama_decode`) goes to `prompt_time`, same as text chunks. Bitmap loading and tokenization are timed by neither, so they fall only in `ttft`; `ttft ≈ media_time + prompt_time` up to that overhead.
- **QAIRT** — `media_time` is the encoder wall time (`encodeVision`); the media-token NPU prefill stays in `prompt_time` (= `ttft − media_time`). QAIRT's `prompt_time` is derived from `ttft`, so treat it as indicative, not exact. The encoder number itself is directly measured.
