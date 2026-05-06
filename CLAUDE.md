# CLAUDE.md

## Project

Multi-platform AI inference runtime (Snapdragon / Hexagon focus).
Languages: C/C++ (SDK), Go (CLI), Python (bindings), Java/JNI (Android).
Build systems: Bazel (CLI) + CMake (SDK).

## Hard constraints

- **Never move or reuse a published git tag.** If the wrong tag shipped, cut a higher one.
- Do not modify third-party code.
- **Follow [CONTRIBUTING.md](CONTRIBUTING.md)** for branch naming, commit / PR title format, pre-commit checks, and the FFI-update rule when changing public SDK headers.
- **Device mapping lives in the SDK: `sdk/src/device.cpp` (`geniex_resolve_device`).** CLI, pybind, and Android are all thin wrappers over that one function — do not re-implement the alias table in a binding. Aliases: `cpu` / `gpu` / `npu` / `hybrid`, with `npu` = pin `HTP0` and `hybrid` = empty `device_id` + `n_gpu_layers=999` (llama.cpp's per-tensor HTP+CPU scheduler, the fast path on Snapdragon). Default when the user passes nothing: `hybrid` for `llama_cpp`, `npu` for `qairt`. QAIRT is NPU-only — other aliases are coerced with a warning, never an error. Model-specific overrides: families that can't run on hybrid (currently anything matching `gpt-oss`) default to `npu` on `llama_cpp` — `is_llama_cpp_hybrid_incompatible` in `sdk/src/device.cpp`. When changing the table, rebuild the SDK bridge (`/build`), then re-sync the table in [notes/run.md § Device mapping](notes/run.md#device-mapping). Any change to `geniex.h` is an FFI change — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Workflows

- Build anything (CLI / SDK bridge / release installer) → run `/build`.
- Cut a release / bump the version → run `/release`.
- Onboarding the AI setup itself → see [notes/AI.md](notes/AI.md).
