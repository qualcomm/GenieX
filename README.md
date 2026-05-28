# Geniex

Multi-platform AI inference runtime for Snapdragon / Hexagon — runs LLMs on NPU, GPU, or CPU through a pluggable C SDK with Go (CLI), Python, and Java (Android) bindings.

> Status: pre-1.0, under active development. Public API and tags may change; see [notes/release.md](notes/release.md).

## Runtimes & compute units

| Runtime / GGML backend | Compute unit   | Model format | Enabled by                                |
| ---------------------- | -------------- | ------------ | ----------------------------------------- |
| llama.cpp / Hexagon    | Snapdragon NPU | GGUF         | `llama_cpp` + `-DGGML_HEXAGON=ON`         |
| llama.cpp / OpenCL     | Adreno GPU     | GGUF         | `llama_cpp` + `-DGGML_OPENCL=ON`          |
| QAIRT / QNN            | Snapdragon NPU | QAIRT `.bin` | `qairt` + `-DGENIEX_PLUGIN_QAIRT=ON`      |
| llama.cpp / CPU        | Any            | GGUF         | `llama_cpp` (default; disable both flags) |

The `llama_cpp` and `qairt` runtimes both target the NPU but through **separate user-space stacks** (ggml-hexagon DSP skels vs. Qualcomm QNN) that consume **different model formats**. They are not substitutes. QAIRT libs are bundled under `third-party/geniex-qairt/`; Hexagon and OpenCL SDKs are external installs.

## Install

Release assets live on the [Releases page](https://github.com/qcom-ai-hub/geniex/releases). `<TAG>` below is the release tag (e.g. `v0.4.0`).

### Windows (installer)

Download `geniex-cli-setup-windows-arm64-<TAG>.exe` and the matching `geniex-sdk-windows-arm64-<TAG>.zip`, then run the installer. For the latest stable installer without picking a tag, fetch [`https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-cli.exe`](https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-cli.exe) — this S3 object always mirrors the newest stable release.

If the SDK name ends in `-selfsigned`, first follow [notes/run.md § Self-signed fallback](notes/run.md#self-signed-fallback) to import `ggml-htp-v1.cer` and enable test-signing. Full walkthrough: [notes/run.md § Running a prebuilt CI release](notes/run.md#running-a-prebuilt-ci-release-windows-on-snapdragon).

### Linux ARM64

Install on a Snapdragon device (EVK, container, or any ARM64 Linux with a Qualcomm BSP):

```bash
# Optional: verify QCOM driver and system-library prerequisites first.
curl -fsSL https://raw.githubusercontent.com/qcom-ai-hub/geniex/main/cli/release/linux/check.sh | sh

curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh
```

If the launcher's directory isn't on your `PATH` yet, the installer prints the exact line to add — typically:

```bash
echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc   # or ~/.zshrc / ~/.profile
```

Open a new shell or `source` that file, then use it:

```bash
geniex pull Qwen/Qwen3-0.6B-GGUF
geniex infer Qwen/Qwen3-0.6B-GGUF -p "Hello, in one short sentence please."
```

Pin a version: `... | sh -s -- --version v0.1.8`. Override the install location: `... | sh -s -- --prefix /opt/geniex`. Other flags: `-q`, `--help`.

---

Prefer Docker (versioned image, repeatable, no host-side install):

```bash
docker pull ghcr.io/qcom-ai-hub/geniex-cli:<TAG>

# interactive mode
docker run -it --rm --privileged \
  -v "$PWD/data:/data" \
  -v /usr/lib:/opt/qcom-lib:ro \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  infer Qwen/Qwen3-0.6B-GGUF

# server mode
docker run -it --rm --privileged \
  -v "$PWD/data:/data" \
  -v /usr/lib:/opt/qcom-lib:ro \
  --network=host \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  serve
# interactive shell connect to server
docker run -it --rm --privileged \
  -v "$PWD/data:/data" \
  -v /usr/lib:/opt/qcom-lib:ro \
  --network=host \
  ghcr.io/qcom-ai-hub/geniex-cli:<TAG> \
  run <model>
```

`--privileged` exposes the NPU/GPU devices and `./data` persists the model cache. `:latest` tracks the most recent stable tag.

### Python

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex
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

Extract `geniex-sdk-<os>-arm64-<TAG>.zip` and point your build at its `include/` and `lib/` directories. To build the SDK in-tree instead, see [notes/build.md § Build the SDK](notes/build.md#build-the-sdk).

## Documentation
To use `geniex`, please refer to [docs](docs/README.md) for detailed guides and API references.


For contribution to this project, see docs below to build and test your changes.

| File                               | Topic                                                                 |
| ---------------------------------- | --------------------------------------------------------------------- |
| [notes/build.md](notes/build.md)     | Build CLI, SDK, and Python bindings (Linux / Windows ARM64 / Android) |
| [notes/run.md](notes/run.md)         | Runtime / compute-unit selection, model pull, Windows self-signed HTP fallback |
| [notes/release.md](notes/release.md) | SemVer tag procedure, channels, Hexagon HTP signing pipeline          |
| [notes/AI.md](notes/AI.md)           | Claude Code integration (slash commands, skills)                      |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Commits, branches, PR format, FFI-update rule                         |

## License

Apache 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
