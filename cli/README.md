## GenieX CLI

Command-line interface for running AI models locally on **Qualcomm** chipsets. Interfaces with the GenieX core runtime and supports two inference backends: **QAIRT** and **llama.cpp**.

### Logging

`GENIEX_LOG` controls log output across the CLI, the C/C++ SDK, and all language bindings (Go, Python, Android):

| Value   | Emits                                    |
|---------|------------------------------------------|
| `none`  | nothing                                  |
| `error` | errors only                              |
| `warn`  | warnings + errors                        |
| `info`  | info + warnings + errors (**default**)   |
| `debug` | debug + info + warnings + errors         |
| `trace` | everything (requires a debug build)      |

```bash
export GENIEX_LOG="debug"          # bash / zsh
$env:GENIEX_LOG="debug"            # PowerShell
```

`NO_COLOR=1` disables ANSI colors.

### Sliding window (qairt only)

The `qairt` backend has a fixed context length (e.g. 4096 tokens). By default, once the accumulated
conversation history plus a new prompt exceeds it, `geniex infer` returns an out-of-context error and
the session cannot continue.

Pass `--sliding-window` to opt into evicting the oldest tokens (above a small anchored prefix) instead,
letting the conversation continue past the context limit:

```bash
geniex infer <model> --sliding-window
```

This sets `sliding_window: true` on the generation config for every `generate()` call; `llama_cpp`
ignores it (it always context-shifts). Without the flag, exceeding the context length still returns
the original error — the feature is strictly opt-in.

### Model pull

Pull a model non-interactively:

```bash
geniex pull <model>[:<precision>] --model-type <model-type>
```

Pull from a specific model hub:

```bash
geniex pull <model>
geniex pull <model> --model-hub aihub   # options: aihub, hf, localfs
```

Import a model from the local filesystem:

```bash
# hf download <model> --local-dir /path/to/modeldir
geniex pull <model> --model-hub localfs --local-path /path/to/modeldir
```
