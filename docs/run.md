# Model Prepare

Currently we must manually prepare the modelfiles.

Currently `OpenCL` and `Hexagon` backend is support on Windows arm64.

By default, `llama_cpp` use both at same time, if you want to specifiedly use one of them, you can change `DeviceId` in `geniex.manifest` file.

PS: still issue when manually set `DeviceId` to `HTP0`, npu usage is zero.
