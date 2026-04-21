# Model Prepare

Currently we must manually prepare the modelfiles.

Currently `OpenCL` and `Hexagon` backend is support on Windows arm64.

By default, `llama_cpp` use both at same time, if you want to specifiedly use one of them, you can change `DeviceId` in `geniex.json` file.

PS: still issue when manually set `DeviceId` to `HTP0`, npu usage is zero.

# Run

`qairt` models need `geniex.json` to work.

for example, [granite4_micro](https://huggingface.co/yichqian/geniex-qairt-models/blob/main/granite4_micro/geniex.json) model's `geniex.json` is like this:

## build and run local

1. Download model, `hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models`
2. Import model, `bazel run //cli -- pull local/granite4_micro --model-hub localfs --local-path /absolute/path/to/geniex-qairt-models/granite4_micro`
3. Run, `bazel run //cli -- infer local/granite4_micro`

## build from others

For builder:

1. run `bazel build //cli:artifact`
2. export `bazel-bin/cli/artifact.zip` and `ggml-htp-v1.cer`.

For users:

1. Get the artifact from builder and unzip it.
2. Download model, `hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models`
3. run `./geniex.exe pull local/granite4_micro --model-hub localfs --local-path /absolute/path/to/geniex-qairt-models/granite4_micro`
4. run `./geniex.exe infer local/granite4_micro`

## download prebuilt from CI (Windows on Snapdragon)

Every `v*` tag publishes the Windows ARM64 installer on the GitHub Releases page.

### 1. Download

Open `https://github.com/qcom-ai-hub/geniex/releases` and grab:

- `geniex-cli-setup.exe` — the installer
- `ggml-htp-v1.cer` — HTP ops-library signing certificate (required for NPU)

### 2. Install the HTP signing certificate (required before first NPU run)

The `llama_cpp` Hexagon backend loads `libggml-htp.cat`, which is signed by a self-signed test cert. Windows refuses to load it until you both enable test signing **and** trust the cert.

1. **Enable test signing** (elevated PowerShell, then reboot):
   ```powershell
   bcdedit /set TESTSIGNING ON
   ```
   If the command fails with a Secure Boot error, disable Secure Boot in UEFI first, then retry.

2. **Import `ggml-htp-v1.cer` into two stores** using `certlm.msc` (Local Machine certificates):
   - `Trusted Root Certification Authorities` → `Certificates` → right-click → **All Tasks → Import…** → select `ggml-htp-v1.cer`.
   - Repeat the import into `Trusted Publishers` → `Certificates`.

   Both stores are required: Root makes the chain valid, Trusted Publishers suppresses the driver-load prompt.

3. Reboot if you haven't already. Verify with:
   ```powershell
   bcdedit /enum | findstr testsigning   # should show "testsigning   Yes"
   ```

Background / original upstream instructions: `third-party/llama.cpp/docs/backend/snapdragon/windows.md`.

### 3. Run

1. Install with `geniex-cli-setup.exe`.
2. Download a model: `hf download yichqian/geniex-qairt-models --local-dir=geniex-qairt-models`
3. `geniex.exe pull local/granite4_micro --model-hub localfs --local-path <abs-path>\geniex-qairt-models\granite4_micro`
4. `geniex.exe infer local/granite4_micro`

with `llama_cpp` backend, need on extra step:

- Get builder's certificate, `ggml-htp-v1.cer`.
- follow [document](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/windows.md#enable-npu-driver-test-signatures)
  - disable secure boot
  - enable test signing
  - _skip_ create certificate
  - import certificate with `certlm`
