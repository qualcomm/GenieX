## GenieX CLI

GenieX-CLI is a command-line interface tool for running AI models locally on **Qualcomm** chipsets. It interfaces with the GenieX core runtime and supports two inference backends: **QAIRT** and **llama.cpp**.

Enable debug log

```
$env:GENIEX_LOG="debug" # powershell

export GENIEX_LOG="debug" # bash
```

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
