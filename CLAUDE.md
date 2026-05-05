# CLAUDE.md

## Project

Multi-platform AI inference runtime (Snapdragon / Hexagon focus).
Languages: C/C++ (SDK), Go (CLI), Python (bindings), Java/JNI (Android).
Build systems: Bazel (CLI) + CMake (SDK).

## Hard constraints

- **Never move or reuse a published git tag.** If the wrong tag shipped, cut a higher one.
- Do not modify third-party code.
- **Follow [CONTRIBUTING.md](CONTRIBUTING.md)** for branch naming, commit / PR title format, pre-commit checks, and the FFI-update rule when changing public SDK headers.
- **`llama_cpp` NPU selection: leave `device_id` null, never pin to `"HTP0"`.** Setting `device_id` makes the plugin call `mpar.devices = {HTP0}`, which disables llama.cpp's per-tensor hybrid dispatch (HTP for supported ops, CPU for fallbacks) and tanks performance by ~30–50%. For `--device npu` / `device_map="npu"`, pass `device_id=""` + `n_gpu_layers=999` and let llama.cpp place layers. Only set an explicit `"HTP0"` / `"HTP0,HTP1,..."` when the user has typed that literal id — see [notes/run.md § NPU device selection](notes/run.md#npu-device-selection-llama_cpp).

## Workflows

- Build anything (CLI / SDK bridge / release installer) → run `/build`.
- Cut a release / bump the version → run `/release`.
- Onboarding the AI setup itself → see [notes/AI.md](notes/AI.md).
