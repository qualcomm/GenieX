# Run

Terminology used throughout this doc:

- **Runtime** — `llama_cpp` or `qairt` (the `plugin_id`).
- **Compute unit** — NPU, GPU, CPU, or `hybrid` (mapped from `-c/--compute` / `device_id`).
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

Defaults when the user passes nothing (`--compute ""` / `device_map="auto"`):
`npu` for both `llama_cpp` and `qairt`. QAIRT exposes only one device,
so `cpu` / `gpu` / `hybrid` against a qairt model get coerced to `NPU`
with a warning on stderr — the CLI does **not** exit early.

Beyond the aliases, `--compute` also accepts an explicit device list of
concrete ids (`HTP0,HTP1,HTP2,HTP3`, `GPUOpenCL`) — `llama_cpp` passes it
through to llama.cpp verbatim (handy for multi-DSP recipes that need more
than the single `HTP0` the `npu` alias pins); `--ngl` still applies. qairt
is NPU-only, so a device list gets coerced to `NPU` with a warning.

## Power mode

A single knob for both runtimes' HTP DCVS/HMX power and clock management, set via
`geniex_ModelConfig.power_mode` (`--power-mode` on `geniex infer` / `run` / `serve` and
`geniex-bench`; a JSON `power_mode` field on `geniex serve` requests). Go and Python
resolve the alias strings themselves; C/C++ callers that build a `geniex_ModelConfig`
directly (benchmark, examples, JNI) use the header-only table in
[`sdk/include/power_mode_alias.h`](../sdk/include/power_mode_alias.h).

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

- **`llama_cpp`** — upstream's HTP power-mode table plus a GenieX setter
  (`sdk/patches/llama-hexagon-power-mode-setter.patch`; see the comments in
  [`sdk/CMakeLists.txt`](../sdk/CMakeLists.txt)). Only affects sessions
  created after the call — an already-open HTP session (another model still
  loaded in the same process) keeps its old mode until released and
  reacquired; the plugin logs a warning rather than silently no-op'ing.
- **`qairt`** — sets `ModelConfig::perf_profile`, which the QAIRT core (a
  real HTP `PerfProfile` → `QnnHtpPerfInfrastructure` vote) already
  supports. **Precedence:** `--power-mode` wins over anything the model
  bundle's `htp_backend_ext_config.json` sets (`resolveHtpPerfConfig` in
  `geniex-qairt` core: caller > bundle json > default), since
  [`geniex-qairt-plugin#55`](https://github.com/qualcomm/geniex-qairt-plugin/pull/55).

## Compute-unit selection (llama_cpp)

The compute unit is driven by two inputs on `geniex_LlmCreateInput`:

- `device_id` — `HTP0`, `HTP0,HTP1,…`, `GPUOpenCL`, or empty.
- `config.n_gpu_layers` — layers to offload; `-1` = all, `0` = CPU.

[`sdk/plugins/llama_cpp/src/params.cpp`](../sdk/plugins/llama_cpp/src/params.cpp) (`classify_device`,
`resolve_devices`) turns those into one of CPU / GPU / NPU and picks per-platform defaults for
`n_ubatch`, flash-attention, and CPU threading from small tables in the same file. Empty
`device_id` with layers offloaded (`hybrid`) is classified as NPU. Edit those tables rather than
special-casing call sites.

```bash
geniex infer unsloth/Qwen3-1.7B-GGUF:Q4_0                # npu (default)
geniex infer unsloth/Qwen3-1.7B-GGUF:Q4_0 -c gpu
geniex infer unsloth/Qwen3-1.7B-GGUF:Q4_0 -c cpu
geniex infer unsloth/Qwen3-1.7B-GGUF:Q4_0 -c HTP0,HTP1   # explicit multi-session list
```

### Checking which path actually ran

The SDK's default log handler writes to stderr, but the CLI defaults to `--log none`. Run with
`--log info` (or `GENIEX_LOG=info`) and look for the device lines (`Found device: HTP0`,
`Using N device(s)`) and `[Optimise] threadpool attached: …`.

If you see `Device '…' not found, skipping`, the runtime loaded but the GGML backend DLL did not.
Check that test-signing is on (for HTP) or that `ggml-opencl.dll` is present in
`sdk/pkg-geniex/lib/llama_cpp/`.

> Q4_K_M is a suboptimal quant for HTP; some tensors fall back to CPU. Use Q4_0 / Q8_0 for a clean NPU run.

## Running QAIRT models

QAIRT exposes only its Hexagon NPU compute unit (`plugin_id="qairt"`, `device_id="NPU"`). The SDK's `geniex_resolve_device` coerces `-c cpu` / `gpu` / `hybrid` to `npu` with a stderr warning, so existing shell pipelines don't break. Expect a line like:

```
Warning: qairt plugin only supports NPU inference; ignoring device='cpu' and running on NPU
```

Each QAIRT model directory carries a `geniex.json`. The model manager writes one on import when the bundle doesn't ship it.

### Using a custom QNN library

A QAIRT runtime ships with GenieX and is used by default. To run against a different one
without reinstalling, point the plugin at it:

```bash
# via the CLI flag (qairt models only)
geniex infer <qairt-model> --qairt-lib /path/to/qairt/2.XX.0

# or via the environment variable (picked up by any front-end: CLI, pybind, Android)
GENIEX_QAIRT_LIB=/path/to/qairt/2.XX.0 geniex infer <qairt-model>
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

<details><summary>Which runtimes are accepted</summary>

The plugin reaches QNN through the versioned C interface with a load-time floor of
`kMinApiMinor = 27` (QNN C API 2.27), independent of the headers it was compiled against, so
older and newer QAIRT runtimes load without ABI breaks. `QnnHtpNetRunExtensions` is not
needed: the plugin applies `htp_backend_ext_config.json` itself.

An unusable path fails model load immediately and names its source, e.g.
`geniex_set_qairt_runtime_path does not contain QnnHtp.dll (looked in the folder itself and
lib/aarch64-windows-msvc): <path>`.

**One runtime per process.** `QnnHtp` stays loaded for the life of the process, and
`ADSP_LIBRARY_PATH` / `SetDllDirectory` are process-wide. The path is locked at `geniex_init`;
setting it later returns `GENIEX_ERROR_COMMON_ALREADY_INITIALIZED`, and `geniex_deinit` doesn't
unlock it. Comparing runtimes means one process per runtime.

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

### Run a QAIRT bundle

AI Hub bundles pull natively: `geniex pull ai-hub-models/<model>:<precision>` (the chipset is detected,
or set it with `geniex config set chipset`). A bundle you already have on disk imports with
`geniex pull <name> --local-path <dir-or-zip>`. A local build runs the same way through
`bazelisk run //cli -- <args>`; to hand a build to another machine, ship `bazel-bin/cli/artifact.zip`
(plus `ggml-htp-v1.cer` for a self-signed build).

## Running a prebuilt CI release (Windows on Snapdragon)

Every `v*` tag publishes `geniex-cli-setup-windows-arm64-<tag>.exe` on [the Releases page](https://github.com/qualcomm/GenieX/releases). The installer bundles the SDK. The SDK zip name tells you how the HTP catalog is signed:

| Filename                                        | HTP signing        | Extra setup                                 |
|-------------------------------------------------|--------------------|---------------------------------------------|
| `geniex-sdk-windows-arm64-<tag>.zip`            | Microsoft-signed   | None.                                       |
| `geniex-sdk-windows-arm64-<tag>-selfsigned.zip` | Self-signed (test) | See [Self-signed fallback](#self-signed-fallback). |

A release that also attaches `ggml-htp-v1.cer` is self-signed.

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
