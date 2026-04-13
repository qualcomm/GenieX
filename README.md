## Directory Structure

- All libraries and tools intended for external release are placed in the repository root
- Dependencies are managed through the build system rather than by directory layout
- Use a single unified build system to avoid complexity caused by different subprojects using different build tools

## Bazel

- Advantages
  - Supports C++/Go/Python/Android and can package Docker images
  - Integrates lint/test/build, so CI only needs to run Bazel commands; local development can also use Bazel to maintain consistency
  - Uses a self-maintained toolchain, unaffected by the system environment
  - Easier to integrate custom build rules compared to CMake
- Disadvantages
  - May be overly complex for small projects (but our project spans multiple languages and platforms)
  - Steeper learning curve (there are plenty of resources; AI can help)

### Common Commands

- `bazel query //...` list all targets in the workspace
- `bazel (build|run) --define=VERSION=XXXX [args]` build/run with version string
- `bazel run [args]` build and run the CLI target, for example: `bazel run infer Qwen/Qwen3-0.6B-GGUF` is equal to `bazel run //cli/cmd/geniex -- infer Qwen/Qwen3-0.6B-GGUF`

#### go tips

- `bazel run //gazelle` update BUILD files from `go.mod`
- `# gazelle:resolve go ://example.com //local/foo:go_default_library` resolve external go module to local bazel target

## Code Structure

```
.
├── .github/                  # GitHub workflows and configs
│   ├── actions/              # reusable actions, like env setup and s3 config
│   │   ├── env.yml
│   │   └── s3.yml
│   ├── scripts/              # reusable scripts used by workflows, like create github release
│   │   └── release.js
│   └── workflows/            # github actions workflows, it's simple because we use bazel for everything
│       ├── build.yml
│       ├── lint.yml
│       └── test.yml
│
├── .vscode/                  # VS Code debug settings, AI agent rules, LLM prompts
│
├── sdk/                      # C API layer (entry for CLI and bindings)
│   ├── include/              # public C API headers
│   ├── libs/                 # resource .so files (QAIRT, Hexagon, utility libs)
│   ├── src/                  # C API source, plugin loading, common utilities
│   └── BUILD.bazel
│
├── third-party/              # third-party dependencies
│   ├── geniex-proc/          # preprocessing and postprocessing repo
│   ├── geniex-qairt/         # core runtime
│   ├── llama.cpp/            # public version, not Qualcomm internal
│   ├── pybind11/             # for Python binding
│   └── jni/                  # for Java binding
│
├── bindings/                 # language bindings and packaging
│   ├── python/               # pybind11 code and setup.py for Python package
│   ├── android/              # JNI code and Maven files for Java package
│   └── docker/               # Dockerfile and scripts for Docker build/release
│
├── cli/                      # command-line interface
│   ├── main.go               # CLI entry point
│   ├── server/               # CLI server components
│   ├── go.mod
│   └── go.sum
│
├── docs/                     # documentation (C API, CLI, Python, Maven, Docker)
│
├── scripts/                  # build, release, signing, file upload/download scripts
│
├── tests/                    # unit/integration tests for C API, Python, Java
│   ├── qdc/                  # QDC device connection/testing scripts/configs
│   ├── include/              # test headers
│   └── src/                  # test source code
│
├── BUILD.bazel               # root BUILD file
├── MODULE.bazel              # root MODULE file
├── MODULE.bazel.lock
├── LICENSE
└── README.md
```

## Notes

- All tutorials, cookbook and sample apps are in a separate repo [geniex-app](https://github.com/geniex-app).
- Should we keep `dlopen` plugins? There are only two plugins now, and both exist on windows/linux arm64.
- Since we will open source all the code, maybe we can static link everything for simplicity.
