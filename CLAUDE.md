# CLAUDE.md

## Project

Multi-platform AI inference runtime (Snapdragon / Hexagon focus).
Languages: C/C++ (SDK), Go (CLI), Python (bindings), Java/JNI (Android).
Build systems: Bazel (CLI) + CMake (SDK).

## Hard constraints

- **Never move or reuse a published git tag.** If the wrong tag shipped, cut a higher one.
- Do not modify third-party code.
- **Follow [CONTRIBUTING.md](CONTRIBUTING.md)** for branch naming, commit / PR title format, pre-commit checks, and the FFI-update rule when changing public SDK headers.
- **Compute-unit alias mapping lives in the SDK: `sdk/src/device.cpp` (`geniex_resolve_device`).** CLI, pybind, and Android are all thin wrappers over that one function — do not re-implement the alias table in a binding. Aliases: `cpu` / `gpu` / `npu` / `hybrid`, with `npu` = pin `HTP0` and `hybrid` = empty `device_id` (llama.cpp's per-tensor HTP+CPU scheduler, the fast path on Snapdragon). `n_gpu_layers` passes through the SDK unchanged for llama_cpp gpu / npu / hybrid; the CLI/binding default is `-1`, which llama.cpp reads as "all layers". `cpu` forces it to 0, and qairt forces it to 0 (its plugin rejects any non-zero value). Default when the user passes nothing: `npu` for both `llama_cpp` and `qairt`. QAIRT is NPU-only — other aliases are coerced with a warning, never an error. When changing the table, rebuild the SDK bridge (`/build`), then re-sync the table in [notes/run.md § Compute-unit aliases](notes/run.md#compute-unit-aliases). Any change to `geniex.h` is an FFI change — see [CONTRIBUTING.md](CONTRIBUTING.md).
- **Power-mode alias mapping lives in the SDK: `sdk/src/device.cpp` (`geniex_resolve_power_mode`), same one-source-of-truth rule as the compute-unit table.** `geniex_ModelConfig.power_mode` (`--power-mode`) is a single HTP DCVS/HMX power/clock-management knob shared by `qairt` and `llama_cpp`; empty / `default` resolves to `burst` on both, matching pre-existing behavior. `llama_cpp` support rides on an unmerged internal PR carried as `sdk/patches/llama-hexagon-power-mode.patch` + `llama-hexagon-power-mode-setter.patch` — do not bump the `third-party/llama.cpp` pin past that PR's merge without refreshing/removing the first patch. `qairt` sets `ModelConfig::perf_profile`, which a bundle's `htp_backend_ext_config.json` can still override. See [notes/run.md § Power mode](notes/run.md#power-mode).

## Workflows

- Build anything (CLI / SDK bridge / release installer) → run `/build`.
- Cut a release / bump the version → run `/release`.
- Onboarding the AI setup itself → see [notes/AI.md](notes/AI.md).
