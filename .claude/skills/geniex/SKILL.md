---
name: geniex
description: Use the geniex CLI — pull, list, infer, serve, benchmark models on Snapdragon CPU/GPU/NPU — and debug what actually ran. Use when running geniex commands, reproducing a user report, or verifying a change end to end.
---

# Using geniex

User reference: [docs/en/run/cli/reference.mdx](../../../docs/en/run/cli/reference.mdx). `geniex <cmd> --help` is authoritative for flags.

## Model ids

`<name>[:<precision>]`, e.g. `unsloth/Qwen3-0.6B-GGUF:Q4_0`. Hub auto-routing:

| Id | Source |
|---|---|
| `org/repo` | Hugging Face |
| bare name, `qualcomm/*`, `ai-hub-models/*` | Qualcomm AI Hub (qairt bundles) |
| `https://modelscope.cn/models/...` | ModelScope (bare `org/repo` is never ModelScope; use `--model-hub modelscope`) |
| `docker.io/...` | Docker Hub |
| `--local-path <dir or .zip>` | local import |

## Non-interactive use (scripts, CI, assistants)

- **Always give the precision.** Without it, `pull` opens a picker (or silently takes the top candidate without a TTY), and `infer` asks when several are cached.
- **`infer` / `decide` auto-pull** a missing model. **`serve` never does**: `geniex pull` first.
- One-shot prompt: `geniex infer <id>:<prec> -c npu -p "<prompt>"`. Don't pipe stdin.
- `geniex ls` shows what's cached and the precisions you can name. Models live in `<data-dir>/models` (default `~/.cache/geniex`, or `--data-dir` / `GENIEX_DATADIR`).

## Compute

`-c cpu|gpu|npu|hybrid|auto` or an explicit list like `HTP0,HTP1`. Empty/`auto` means **npu** for both runtimes. qairt bundles are NPU-only: other values are coerced to NPU with a warning, not an error. `-n` (ngl) defaults to -1 = all layers.

## Debugging

- `--log debug` or `GENIEX_LOG=trace` prints CLI and SDK logs to stderr. Check which device and plugin actually loaded before trusting a number.
- **Testing a local build**: an installed release CLI on PATH (Windows `%LOCALAPPDATA%\GenieX CLI\`) can shadow your build's `geniex.dll` and plugins. Put `sdk/pkg-geniex/lib` first on PATH and set `GENIEX_PLUGIN_PATH` to it, then confirm the plugin path in the trace log.
- Downloads blocked (e.g. no direct access to huggingface.co): set `HF_ENDPOINT` to a mirror, or `MODELSCOPE_ENDPOINT`, and retry `geniex pull`. Gated HF repos: set `GENIEX_HFTOKEN`.
- Windows HTP "failed to open session 0x80000406" on a self-signed build: import `ggml-htp-v1.cer` ([notes/run.md](../../../notes/run.md) § Self-signed fallback).

## Benchmark

`geniex-bench` is the measurement tool, not `geniex infer`: [docs/en/tutorials/benchmarking.mdx](../../../docs/en/tutorials/benchmarking.mdx). QDC CI benches: [notes/bench.md](../../../notes/bench.md).
