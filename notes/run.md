# Run

Two plugins ship with geniex and both can drive the Snapdragon NPU, but through **separate user-space stacks** that consume **different model formats**:

- **`llama_cpp`** — GGUF models, targets Hexagon NPU (via `ggml-hexagon`), Adreno GPU (via OpenCL), or CPU.
- **`qairt`** — QAIRT `.bin` shards, targets Hexagon NPU via Qualcomm's QNN runtime.

They are not interchangeable; the plugin is chosen per model.

> For the CI/S3 signing pipeline that backs HTP releases, see [release.md § Hexagon HTP signing](release.md#hexagon-htp-signing).

## Device mapping

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
| `cpu`    | empty                   | `0`                     | Pure CPU.                                                                                   |
| `gpu`    | `GPUOpenCL`             | (none; caller default)  | Adreno via OpenCL. Pair with a high `--ngl`.                                               |
| `npu`    | `HTP0`                  | (none; caller default)  | Pinned single-session HTP. Deterministic, slower on LLMs — see § NPU device selection.      |
| `hybrid` | empty                   | `999`                   | `llama_cpp` fast path: per-tensor HTP+CPU scheduler. Default when nothing is passed.         |

Defaults when the user passes nothing (`--device ""` / `device_map="auto"`):
`hybrid` for `llama_cpp`, `npu` for `qairt`. QAIRT exposes only one
device, so `cpu` / `gpu` / `hybrid` against a qairt model get coerced to
`NPU` with a warning on stderr — the CLI does **not** exit early.

**Model-specific default override**: the SDK also inspects the model
name when the caller passes no device. Families listed in
`is_llama_cpp_hybrid_incompatible` ([`sdk/src/device.cpp`](../sdk/src/device.cpp))
— currently anything whose name contains `gpt-oss` — default to `npu`
instead of `hybrid`, because the per-tensor hybrid scheduler can't
place all of their ops on HTP end-to-end. Pass `--device hybrid`
explicitly to override the override. Adding a new family means editing
that one function and rebuilding the SDK bridge.

Concrete ids (`HTP0,HTP1,HTP2,HTP3`, `GPUOpenCL`, etc.) pass through
unchanged when supplied via `<plugin>:<device>` or manifest `DeviceId`.

## Backend selection (llama_cpp)

`llama_cpp` supports OpenCL and Hexagon on Windows ARM64. The backend is driven by two inputs on `geniex_LlmCreateInput`:

- `device_id` — string, plugin-specific (`HTP0`, `GPUOpenCL`, `CPU`, …).
- `config.n_gpu_layers` — int; how many layers to offload. `999` = all.

### NPU device selection (llama_cpp)

`sdk/plugins/llama_cpp/src/llm.cpp:73-114` branches on whether `device_id` is non-null, producing **two runtime paths** with very different performance:

1. **`device_id` null + `n_gpu_layers=999`** (the `hybrid` alias) → llama.cpp's **per-tensor scheduler**. It inspects each tensor and assigns it to whichever registered backend supports the op (HTP for computable ops, CPU for fallbacks), using CPU-resident buffers for the fallback tensors. **Fast path.** On X1E80100 + Qwen3-1.7B-Q8_0: ~90 tok/s prefill, ~27 tok/s decode, ~200 ms TTFT. Task Manager shows NPU pegged.

2. **`device_id="HTP0"`** (the `npu` alias) → plugin calls `ggml_backend_dev_by_name("HTP0")` and sets `mpar.devices = {HTP0}`. This **pins the model to a single-device layout** and disables per-tensor hybrid assignment. Any op HTP doesn't support gets handled less efficiently. On the same model: ~60 tok/s prefill, ~22 tok/s decode, ~350 ms TTFT. Task Manager shows CPU pegged (the host thread driving HTP busy-waits, *plus* all fallbacks run there). Useful when you want deterministic layout / all weights on a known device.

Bonus: when the `device_id` string starts with `"HTP0"`, the plugin also flips KV cache to Q8_0 and enables flash-attn (`llm.cpp:136-140`). Orthogonal to perf — path (2) is slower than (1) even with those enabled.

**Rule of thumb:** use `--device hybrid` (or leave `--device` empty) for fastest throughput; use `--device npu` when you need determinism or when debugging placement. Pass literal `"HTP0"` / `"HTP0,HTP1,HTP2,HTP3"` via the manifest `DeviceId` when the user explicitly wants that layout.

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

Or override `DeviceId` in the per-model manifest (takes precedence when `--device` is unset and the plugin is not `qairt`):

```powershell
$m = "$env:USERPROFILE\.cache\geniex\models\NexaAI\Qwen3-0.6B-GGUF\geniex.json"
$j = Get-Content $m -Raw | ConvertFrom-Json
$j.DeviceId = ""    # hybrid; or "GPUOpenCL" / "HTP0,HTP1,HTP2,HTP3" / etc.
$j | ConvertTo-Json -Depth 20 | Set-Content $m
```

### Sanity-checking which path actually ran

The SDK's default log handler is a no-op in release builds (`sdk/src/ml.cpp:36-60`), so `stdout`/`stderr` stays silent and "did it actually use HTP?" is easy to guess wrong. Three ways to check:

- **Python:** set `GENIEX_LOG=INFO`. The Python binding installs a `geniex_set_log` callback that routes SDK messages (`Found device: HTP0`, `Using N device(s)`, etc.) to stderr. If you see `Found device: …` lines you're on the **pinned-`HTP0` path** (the `npu` alias); absence = hybrid path.
- **Windows:** Task Manager's NPU graph. Hybrid lights it up; pinned-`HTP0` pegs the CPU (host thread busy-waits HTP the whole inference).
- **Signature:** on Snapdragon X1E80100 + a 1.7B Q8_0 model, hybrid gives prefill ≳ 80 tok/s and TTFT ≲ 250 ms; pinned-`HTP0` gives prefill ≲ 65 tok/s and TTFT ≳ 340 ms. Prefill and TTFT separate the two paths more cleanly than decode.

If you see `Device '…' not found, skipping`, the plugin loaded but the backend DLL did not — verify test-signing is still on (for HTP) or that `ggml-opencl.dll` is present in `sdk/pkg-geniex/lib/llama_cpp/`.

> Q4_K_M is a suboptimal quant for HTP — it prefers Q4_0 / Q8_0, so some tensors fall back to CPU. Use Q4_0 for a clean NPU run.

## Running QAIRT models

QAIRT exposes only its Hexagon NPU device (plugin `qairt`, device_id `NPU`). The SDK's `geniex_resolve_device` coerces `--device cpu` / `gpu` / `hybrid` to `npu` with a stderr warning so existing shell pipelines don't break — expect a line like:

```
Warning: qairt plugin only supports NPU inference; ignoring device='cpu' and running on NPU
```

QAIRT models need a `geniex.json` to work. See the [granite4_micro example](https://huggingface.co/yichqian/geniex-qairt-models/blob/main/granite4_micro/geniex.json).

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

Every `v*` tag publishes the Windows ARM64 installer on [the Releases page](https://github.com/qcom-ai-hub/geniex/releases). Download both:

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
