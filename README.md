# Geniex

Multi-platform AI inference runtime for Snapdragon / Hexagon — runs LLMs on NPU, GPU, or CPU through a pluggable C SDK with Go (CLI), Python, and Java (Android) bindings.

> Status: pre-1.0, under active development. Public API and tags may change; see [docs/release.md](docs/release.md).

## Backends

| Backend        | Hardware              | Model format     | Plugin enabled by                          |
|----------------|-----------------------|------------------|--------------------------------------------|
| Hexagon (HTP)  | Snapdragon NPU        | GGUF             | `llama_cpp` + `-DGGML_HEXAGON=ON`          |
| OpenCL         | Adreno GPU            | GGUF             | `llama_cpp` + `-DGGML_OPENCL=ON`           |
| QAIRT / QNN    | Snapdragon NPU        | QAIRT `.bin`     | `qairt` + `-DGENIEX_PLUGIN_QAIRT=ON`       |
| CPU            | Any                   | GGUF             | `llama_cpp` (default; disable both flags)  |

The `llama_cpp` and `qairt` plugins both target the NPU but through **separate user-space stacks** (ggml-hexagon DSP skels vs. Qualcomm QNN) that consume **different model formats**. They are not substitutes. QAIRT libs are bundled under `third-party/geniex-qairt/`; Hexagon and OpenCL SDKs are external installs.

## Quick Start

Install [Bazelisk](https://github.com/bazelbuild/bazelisk):

- Windows: `winget install --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

Build and run the CLI — all dependencies are fetched by Bazel:

```bash
bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF
```

For Windows ARM64 + NPU builds, Android cross-compilation, Python bindings, and release packaging, see [docs/build.md](docs/build.md). For backend selection, model preparation, and HTP test-signing, see [docs/run.md](docs/run.md).

## Documentation

| File | Topic |
|---|---|
| [docs/build.md](docs/build.md)       | Build CLI, SDK, and Python bindings (Linux / Windows ARM64 / Android) |
| [docs/run.md](docs/run.md)           | Backend selection, model pull, Windows self-signed HTP fallback        |
| [docs/release.md](docs/release.md)   | SemVer tag procedure, channels, Hexagon HTP signing pipeline           |
| [docs/AI.md](docs/AI.md)             | Claude Code integration (slash commands, skills)                       |
| [CONTRIBUTING.md](CONTRIBUTING.md)   | Commits, branches, PR format, FFI-update rule                          |

## Repository layout

| Path            | Contents                                                     |
|-----------------|--------------------------------------------------------------|
| `sdk/`          | C API: public headers, plugin loader, bundled native libs    |
| `cli/`          | Go CLI (entry point + server)                                |
| `bindings/`     | Python (pybind11), Android (JNI), Docker packaging           |
| `third-party/`  | `geniex-qairt`, `geniex-proc`, `llama.cpp`, `pybind11`, `jni` |
| `docs/`         | Developer documentation                                      |
| `examples/`     | Sample apps (Android, ...)                                   |
| `tests/`        | C API / Python / Java tests, QDC device configs              |
| `scripts/`      | Build, release, signing, upload/download                     |

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Related

Tutorials, cookbooks, and sample apps: [github.com/geniex-app](https://github.com/geniex-app).
