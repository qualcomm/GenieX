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

with `llama_cpp` backend, need on extra step:

- Get builder's certificate, `ggml-htp-v1.cer`.
- follow [document](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/windows.md#enable-npu-driver-test-signatures)
  - disable secure boot
  - enable test signing
  - _skip_ create certificate
  - import certificate with `certlm`
