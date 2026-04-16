## 构建

### 环境设置

#### Windows (x64)

安装 MSYS2

- `winget install --id=MSYS2.MSYS2 -e`
- 将 `C:\msys64\usr\bin` 和 `C:\msys64\mingw64\bin` 添加到你的 PATH 中
- `pacman -Syu`
- `pacman -S make mingw-w64-x86_64-gcc`
- 重启终端

配置 GO 环境

```powershell
go env -w CGO_ENABLED=1
```

#### Windows (arm64)

安装 MSYS2

```powershell
winget install --id=MSYS2.MSYS2 -e
```

在 PowerShell 中添加 MSYS2 的 bin 目录到 PATH：

```powershell
[Environment]::SetEnvironmentVariable(
    "PATH",
    $env:PATH + ";C:\msys64\usr\bin;C:\msys64\clangarm64\bin",
    "User"
)
# 关闭并重新打开 PowerShell 以使更改生效
```

打开 MSYS2 shell 并运行：

```bash
pacman -Syu
pacman -S make mingw-w64-clang-aarch64-clang
```

配置 GO 环境

```powershell
go env -w CGO_ENABLED=1
go env -w CC=clang.exe
go env -w CXX=clang++.exe
```

#### MacOS / Linux

通过你的包管理器安装 `make`、`gcc` 或 `clang`。

配置 GO 环境

```bash
go env -w CGO_ENABLED=1
```

### 安装 `geniex-bridge`

有两种方式安装 bridge 库：

1. 从 S3 bucket 下载

```bash
make download
```

2. 从本地文件安装

```bash
make link
```

---

### 构建项目

完成依赖安装和 bridge 库安装后，构建项目：

```bash
make build
```

---

### 运行项目

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

---

### 项目测试

```
pip install psutil

python tests/run.py
```
