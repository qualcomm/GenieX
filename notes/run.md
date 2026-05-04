# Run

Two plugins ship with geniex and both can drive the Snapdragon NPU, but through **separate user-space stacks** that consume **different model formats**:

- **`llama_cpp`** — GGUF models, targets Hexagon NPU (via `ggml-hexagon`), Adreno GPU (via OpenCL), or CPU.
- **`qairt`** — QAIRT `.bin` shards, targets Hexagon NPU via Qualcomm's QNN runtime.

They are not interchangeable; the plugin is chosen per model.

> For the CI/S3 signing pipeline that backs HTP releases, see [release.md § Hexagon HTP signing](release.md#hexagon-htp-signing).

## Backend selection (llama_cpp)

`llama_cpp` supports OpenCL and Hexagon on Windows ARM64, and by default uses both. The CLI has no `--device` flag; the backend is driven by the `DeviceId` field in `%USERPROFILE%\.cache\geniex\models\<name>\geniex.json`. Recognized values (from `sdk/plugins/llama_cpp/src/llm.cpp:73-114`):

| Target      | `DeviceId`                        | Notes                                                                                              |
|-------------|-----------------------------------|----------------------------------------------------------------------------------------------------|
| Hexagon NPU | `HTP0` (or `HTP0,HTP1,HTP2,HTP3`) | Starting with `HTP0` also flips KV cache to Q8_0 + flash-attn ON. Plugin auto-sets `GGML_HEXAGON_NDEV=4`. |
| Adreno GPU  | `GPUOpenCL`                       | Pass `-n 999` on `infer` so all layers go to the OpenCL device.                                    |
| CPU         | `CPU` or empty                    | Default.                                                                                           |

Example — run a local GGUF on the Hexagon NPU (PowerShell):

```powershell
# 1. Register the file (the last path segment becomes the model namespace)
bazelisk run //cli -- pull Qualcomm/Qwen3-4B-Instruct`
  --model-hub localfs `
  --local-path C:\path\to\modelfiles `
  --model-type llm

# 2. Flip DeviceId
$m = "$env:USERPROFILE\.cache\geniex\models\NexaAI\Qwen3-0.6B-GGUF\geniex.json"
$j = Get-Content $m -Raw | ConvertFrom-Json
$j.DeviceId = "HTP0"    # or "GPUOpenCL" or "CPU"
$j | ConvertTo-Json -Depth 20 | Set-Content $m

# 3. Run
bazelisk run //cli -- infer Qualcomm/Qwen3-4B-Instruct -p "Hello"
```

Sanity-check with `-v`: look for `Found device: HTP0` / `Found device: GPUOpenCL`. If you see `Device '…' not found, skipping`, the plugin loaded but the backend DLL did not — verify test-signing is still on (for HTP) or that `ggml-opencl.dll` is present in `sdk/pkg-geniex/lib/llama_cpp/`.

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
