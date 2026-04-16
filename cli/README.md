## GenieX CLI

GenieX-CLI is a command-line interface tool for running AI models locally on **Qualcomm** chipsets. It interfaces with the GenieX core runtime and supports two inference backends: **QAIRT** and **llama.cpp**.

## Build

### Setup Environment

#### Windows (x64)

Setup MSYS2

- `winget install --id=MSYS2.MSYS2 -e`
- add `C:\msys64\usr\bin` and `C:\msys64\mingw64\bin` to your PATH
- `pacman -Syu`
- `pacman -S make mingw-w64-x86_64-gcc`
- restart terminal

Setup GO Env

```powershell
go env -w CGO_ENABLED=1
```

#### Windows (arm64)

Setup MSYS2

```powershell
winget install --id=MSYS2.MSYS2 -e
```

Add MSYS2 bins to PATH (run in PowerShell):

```powershell
[Environment]::SetEnvironmentVariable(
    "PATH",
    $env:PATH + ";C:\msys64\usr\bin;C:\msys64\clangarm64\bin",
    "User"
)
# Close and reopen PowerShell for the change to take effect
```

Open an MSYS2 shell and run:

```bash
pacman -Syu
pacman -S make mingw-w64-clang-aarch64-clang
```

Setup GO Env

```powershell
go env -w CGO_ENABLED=1
go env -w CC=clang.exe
go env -w CXX=clang++.exe
```

#### MacOS / Linux

install `make`, `gcc` or `clang` via your package manager.

Setup GO Env

```bash
go env -w CGO_ENABLED=1
```

### Install `geniex-bridge`

There are two ways to install the bridge library:

1. From S3 bucket

```bash
make download
```

2. From local files

```bash
make link
```

---

### Build Project

Once the prerequisites and bridge library are installed, build the project:

```bash
make build
```

---

### Run Project

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

---

### Test Project

```
pip install psutil

python tests/run.py
```
