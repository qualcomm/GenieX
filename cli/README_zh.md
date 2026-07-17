## GenieX CLI

用于在 **Qualcomm** 芯片上本地运行 AI 模型的命令行工具。对接 GenieX 核心运行时，并支持两种推理后端：**QAIRT** 与 **llama.cpp**。

### 日志

`GENIEX_LOG` 控制 CLI、C/C++ SDK 以及所有语言绑定（Go、Python、Android）的日志输出：

| 取值    | 输出内容                                 |
|---------|------------------------------------------|
| `none`  | 无输出                                   |
| `error` | 仅错误                                   |
| `warn`  | 警告 + 错误                              |
| `info`  | 信息 + 警告 + 错误（**默认**）           |
| `debug` | 调试 + 信息 + 警告 + 错误                |
| `trace` | 全部输出（需要 debug 构建）              |

```bash
export GENIEX_LOG="debug"          # bash / zsh
$env:GENIEX_LOG="debug"            # PowerShell
```

设置 `NO_COLOR=1` 可关闭 ANSI 颜色。

### 滑动窗口（仅 qairt）

`qairt` 后端的上下文长度是固定的（例如 4096 tokens）。默认情况下，一旦累计的对话历史加上新的 prompt 超出该长度，`geniex infer` 会返回超出上下文（out-of-context）错误，会话无法继续。

传入 `--sliding-window` 可选择丢弃最旧的 tokens（保留一小段锚定前缀），从而让对话在超出上下文限制后仍能继续：

```bash
geniex infer <model> --sliding-window
```

该选项会在每次 `generate()` 调用时将 generation config 中的 `sliding_window` 设为 `true`；`llama_cpp` 会忽略该选项（它始终会做 context-shift）。未传入该标志时，超出上下文长度仍会返回原来的错误——该功能为严格的可选开启。

### 模型拉取

以非交互方式拉取模型：

```bash
geniex pull <model>[:<precision>] --model-type <model-type>
```

从指定的模型仓库拉取：

```bash
geniex pull <model>
geniex pull <model> --model-hub aihub   # 可选：aihub、hf、localfs
```

从本地文件系统导入模型：

```bash
# hf download <model> --local-dir /path/to/modeldir
geniex pull <model> --model-hub localfs --local-path /path/to/modeldir
```
