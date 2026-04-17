# Model Prepare

Currently we must manually prepare the modelfiles.

Currently `OpenCL` and `Hexagon` backend is support on Windows arm64.

By default, `llama_cpp` use both at same time, if you want to specifiedly use one of them, you can change `DeviceId` in `geniex.manifest` file.

PS: still issue when manually set `DeviceId` to `HTP0`, npu usage is zero.

# Run

## build local

### Download and Import Model

### Run

`bazel run //cli:geniex_cli`

## build from others

### Get Artifacts

`bazel run //cli:artifact`, then get artifact file at `./bazel-bin/cli/artifact.zip`

### Get and Import Cert

### Download and Import Model

see build local

### Run
