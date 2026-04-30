## GenieX CLI

开启 debug 日志

```
$env:GENIEX_LOG="debug" # powershell

export GENIEX_LOG="debug" # bash
```

拉取模型（非交互模式）

```bash
geniex pull <model>[:<quant>] --model-type <model-type>
```

从 model hub 拉取模型

```bash
geniex pull <model>
geniex pull <model> --model-hub s3 # 指定 model hub，[volces|modelscope|s3|hf]
```

从本地文件系统导入模型

```bash
# hf download <model> --local-dir /path/to/modeldir
geniex pull <model> --model-hub localfs --local-path /path/to/modeldir
```
