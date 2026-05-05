# Run

Two plugins ship with geniex and both can drive the Snapdragon NPU, but through **separate user-space stacks** that consume **different model formats**:

- **`llama_cpp`** — GGUF models, targets Hexagon NPU (via `ggml-hexagon`), Adreno GPU (via OpenCL), or CPU.
- **`qairt`** — QAIRT `.bin` shards, targets Hexagon NPU via Qualcomm's QNN runtime.

They are not interchangeable; the plugin is chosen per model.

> For the CI/S3 signing pipeline that backs HTP releases, see [release.md § Hexagon HTP signing](release.md#hexagon-htp-signing).

## Backend selection (llama_cpp)

`llama_cpp` supports OpenCL and Hexagon on Windows ARM64. The backend is driven by two inputs on `geniex_LlmCreateInput`:

- `device_id` — string, plugin-specific (`HTP0`, `GPUOpenCL`, `CPU`, …).
- `config.n_gpu_layers` — int; how many layers to offload. `999` = all.

### NPU device selection (llama_cpp)

**Critical gotcha: there are two code paths and they behave very differently.**

`sdk/plugins/llama_cpp/src/llm.cpp:73-114` branches on whether `device_id` is non-null:

1. **`device_id` null + `n_gpu_layers=999`** → llama.cpp's **hybrid dispatch**. It inspects each tensor and assigns it to whichever registered backend supports the op (HTP for computable, CPU for fallbacks), using CPU-resident buffers for the fallback tensors. **This is the fast path.** On X1E80100 + Qwen3-1.7B-Q8_0: ~90 tok/s prefill, ~27 tok/s decode, ~200 ms TTFT. Task Manager shows NPU pegged.

2. **`device_id="HTP0"`** → plugin calls `ggml_backend_dev_by_name("HTP0")` and sets `mpar.devices = {HTP0}`. This **pins the model to a single-device layout** and disables per-tensor hybrid assignment. Any op HTP doesn't support gets handled less efficiently. On the same model: ~60 tok/s prefill, ~22 tok/s decode, ~350 ms TTFT. Task Manager shows CPU pegged (because the CPU core driving HTP busy-waits, *plus* all the fallbacks run there). This is the **slow path** and not what the user wants from `--device npu`.

Bonus: when the `device_id` string starts with `"HTP0"`, the plugin also flips KV cache to Q8_0 and enables flash-attn (`llm.cpp:136-140`). That side effect is orthogonal to the perf problem — even with those enabled, path (2) is slower than (1).

**Rule for tooling that wires up `--device npu`:** leave `device_id=""` and set `n_gpu_layers=999`. Only forward a literal `"HTP0"` / `"HTP0,HTP1,HTP2,HTP3"` when the *user* typed that — never synthesize it from a friendly alias.

This regression was introduced in commit `fb98467` (`add device parameter`), which made `resolveDevice()` set `deviceID = "HTP0"` for `--device npu`. First shipped in `v0.1.2-rc.1`, still present in `v0.1.2-rc.2`. The Python binding initially copied the same mistake and was corrected in `bindings/python/geniex/auto.py:_LLAMA_CPP_DEVICE_ALIASES`.

### Device table

| Target      | `device_id` to pass                      | `n_gpu_layers` | Notes                                                                                              |
|-------------|------------------------------------------|----------------|----------------------------------------------------------------------------------------------------|
| Hexagon NPU | **empty**                                | `999`          | Fast hybrid path. Let llama.cpp place tensors across HTP + CPU buffers.                           |
| Adreno GPU  | `GPUOpenCL`                              | `999`          | All layers to OpenCL.                                                                              |
| Pin to HTP  | `HTP0` or `HTP0,HTP1,HTP2,HTP3`          | `999`          | Slow single-device path; also flips KV cache→Q8_0 + flash-attn. Use only if you explicitly need it. |
| CPU         | empty                                    | `0`            | Default.                                                                                           |

### Running from the CLI

Use `--device` (`-d`):

```powershell
geniex infer Qwen/Qwen3-1.7B-GGUF --device npu -p "Hello"
geniex infer Qwen/Qwen3-1.7B-GGUF --device gpu -p "Hello"
geniex infer Qwen/Qwen3-1.7B-GGUF --device cpu -p "Hello"
```

Or override `DeviceId` in the per-model manifest:

```powershell
$m = "$env:USERPROFILE\.cache\geniex\models\NexaAI\Qwen3-0.6B-GGUF\geniex.json"
$j = Get-Content $m -Raw | ConvertFrom-Json
$j.DeviceId = ""    # NPU hybrid; or "GPUOpenCL" / "CPU" / literal "HTP0,HTP1,..."
$j | ConvertTo-Json -Depth 20 | Set-Content $m
```

### Sanity-checking which path actually ran

The SDK's default log handler is a no-op in release builds (`sdk/src/ml.cpp:36-60`), so `stdout`/`stderr` stays silent and "did it actually use HTP?" is easy to guess wrong. Three ways to check:

- **Python:** set `GENIEX_LOG=INFO`. The Python binding installs a `geniex_set_log` callback that routes SDK messages (`Found device: HTP0`, `Using N device(s)`, etc.) to stderr. If you see `Found device: …` lines, you're on the **slow path**; absence = hybrid path.
- **Windows:** Task Manager's NPU graph. Hybrid path lights it up; single-device `HTP0` path pegs the CPU instead (the host thread busy-waits HTP the whole inference).
- **Signature:** on Snapdragon X1E80100 + a 1.7B Q8_0 model, hybrid gives prefill ≳ 80 tok/s and TTFT ≲ 250 ms. Pinned-`HTP0` gives prefill ≲ 65 tok/s and TTFT ≳ 340 ms. Prefill and TTFT separate the two paths more cleanly than decode.

If you see `Device '…' not found, skipping`, the plugin loaded but the backend DLL did not — verify test-signing is still on (for HTP) or that `ggml-opencl.dll` is present in `sdk/pkg-geniex/lib/llama_cpp/`.

> Q4_K_M is a suboptimal quant for HTP — it prefers Q4_0 / Q8_0, so some tensors fall back to CPU. Use Q4_0 for a clean NPU run.

## Running QAIRT models

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
