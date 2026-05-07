## GenieX CLI

GenieX-CLI is a command-line interface tool for running AI models locally on **Qualcomm** chipsets. It interfaces with the GenieX core runtime and supports two inference backends: **QAIRT** and **llama.cpp**.

### Logging

`GENIEX_LOG` is a single environment variable that controls log output across the CLI,
the C/C++ SDK, and all language bindings (Go, Python, Android). Accepted values:

| Value   | Emits                                    |
|---------|------------------------------------------|
| `none`  | nothing                                  |
| `error` | errors only                              |
| `warn`  | warnings + errors                        |
| `info`  | info + warnings + errors (**default**)   |
| `debug` | debug + info + warnings + errors         |
| `trace` | everything (TRACE requires a debug build)|

```
$env:GENIEX_LOG="debug" # powershell

export GENIEX_LOG="debug" # bash
```

`NO_COLOR=1` disables ANSI colors.

Pull model without interactive

```bash
geniex pull <model>[:<quant>] --model-type <model-type>
```

Pull model from model hub

```bash
geniex pull <model>
geniex pull <model> --model-hub s3 # pull from specify model hub, [volces|modelscope|s3|hf]
```

Import model from local filesystem

```bash
# hf download <model> --local-dir /path/to/modeldir
geniex pull <model> --model-hub localfs --local-path /path/to/modeldir
```
