# Geniex

Multi-platform AI inference runtime for Snapdragon / Hexagon — runs LLMs on NPU, GPU, or CPU through a pluggable C SDK with Go (CLI), Python, and Java (Android) bindings.

> Status: pre-1.0, under active development. Public API and tags may change; see [docs/release.md](docs/release.md).

## Backends

| Backend       | Hardware       | Model format | Plugin enabled by                         |
| ------------- | -------------- | ------------ | ----------------------------------------- |
| Hexagon (HTP) | Snapdragon NPU | GGUF         | `llama_cpp` + `-DGGML_HEXAGON=ON`         |
| OpenCL        | Adreno GPU     | GGUF         | `llama_cpp` + `-DGGML_OPENCL=ON`          |
| QAIRT / QNN   | Snapdragon NPU | QAIRT `.bin` | `qairt` + `-DGENIEX_PLUGIN_QAIRT=ON`      |
| CPU           | Any            | GGUF         | `llama_cpp` (default; disable both flags) |

The `llama_cpp` and `qairt` plugins both target the NPU but through **separate user-space stacks** (ggml-hexagon DSP skels vs. Qualcomm QNN) that consume **different model formats**. They are not substitutes. QAIRT libs are bundled under `third-party/geniex-qairt/`; Hexagon and OpenCL SDKs are external installs.

## Install

Release assets live on the [Releases page](https://github.com/qcom-ai-hub/geniex/releases). `<TAG>` below is the release tag (e.g. `v0.4.0`).

### Windows (installer)

Download `geniex-cli-installer-windows-arm64-<TAG>.exe` and the matching `geniex-sdk-windows-arm64-<TAG>.zip`, then run the installer.

If the SDK name ends in `-selfsigned`, first follow [docs/run.md § Self-signed fallback](docs/run.md#self-signed-fallback) to import `ggml-htp-v1.cer` and enable test-signing. Full walkthrough: [docs/run.md § Running a prebuilt CI release](docs/run.md#running-a-prebuilt-ci-release-windows-on-snapdragon).

### Linux (Docker)

```bash
docker pull ghcr.io/qcom-ai-hub/geniex-cli:<TAG>

# interactive mode
docker run -it --rm --privileged -v "$PWD/data:/data" \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  infer Qwen/Qwen3-0.6B-GGUF

# server mode
docker run -it --rm --privileged -v "$PWD/data:/data" \
  --network=host \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  serve
# interactive shell connect to server
docker run -it --rm --privileged -v "$PWD/data:/data" \
  --network=host \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  run <model>
```

`--privileged` + the `./data` bind mount enable NPU device access and model cache persistence. `:latest` tracks the most recent stable tag.

### Python

```bash
pip install --index-url https://test.pypi.org/simple/ geniex
```

The sdist auto-downloads the matching SDK zip per host at install time. API, CLI (`geniex-py`), and env vars: [bindings/python/README.md](bindings/python/README.md). Install sources (GitHub Release URL, offline mirror): [bindings/python/BUILD.md § Install sources](bindings/python/BUILD.md#install-sources).

### Android (AAR)

Download `geniex-android-aar-<TAG>.aar` from the Releases page and reference it as a local dependency:

```kotlin
// settings.gradle.kts
dependencyResolutionManagement {
    repositories { flatDir { dirs("libs") } }
}
// app/build.gradle.kts
dependencies { implementation(files("libs/geniex-android-aar-<TAG>.aar")) }
```

API and architecture: [bindings/android/README.md](bindings/android/README.md).

### SDK zip (integrators)

Extract `geniex-sdk-<os>-arm64-<TAG>.zip` and point your build at its `include/` and `lib/` directories. To build the SDK in-tree instead, see [docs/build.md § Build the SDK](docs/build.md#build-the-sdk).

## Documentation

| File                               | Topic                                                                 |
| ---------------------------------- | --------------------------------------------------------------------- |
| [docs/build.md](docs/build.md)     | Build CLI, SDK, and Python bindings (Linux / Windows ARM64 / Android) |
| [docs/run.md](docs/run.md)         | Backend selection, model pull, Windows self-signed HTP fallback       |
| [docs/release.md](docs/release.md) | SemVer tag procedure, channels, Hexagon HTP signing pipeline          |
| [docs/AI.md](docs/AI.md)           | Claude Code integration (slash commands, skills)                      |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Commits, branches, PR format, FFI-update rule                         |

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
