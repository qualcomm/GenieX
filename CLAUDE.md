# CLAUDE.md

## Project

Multi-platform AI inference runtime (Snapdragon / Hexagon focus).
Languages: C/C++ (SDK), Go (CLI), Python (bindings), Java/JNI (Android).
Build systems: Bazel (CLI) + CMake (SDK).

## Hard constraints

- **Never move or reuse a published git tag.** If the wrong tag shipped, cut a higher one.
- Do not modify third-party code.
- **Before committing code**, run the same lint/format checks CI runs. The authoritative list is [.github/workflows/lint.yml](.github/workflows/lint.yml) — currently `clang-format-20` on touched C/C++ under `sdk/`, `cli/`, `bindings/python/`; `ruff check` + `ruff format --check` on `bindings/python/**/*.py`; `go mod tidy` clean in `cli/`. If CI adds a check, this rule follows — re-read the workflow.
- **When changing public SDK headers under [sdk/include/](sdk/include/)**, the FFI surface in every binding must be updated in the same change, or the binding will crash at load/call time. Bindings to check: [bindings/python/geniex/geniex_sdk/_api.py](bindings/python/geniex/geniex_sdk/_api.py) + [_types.py](bindings/python/geniex/geniex_sdk/_types.py) (ctypes), [bindings/go/](bindings/go/) (cgo), [bindings/android/app/src/main/cpp/](bindings/android/app/src/main/cpp/) (JNI). After updating one, ask the user whether the others also need to change.

## Workflows

- Build anything (CLI / SDK bridge / release installer) → run `/build`.
- Cut a release / bump the version → run `/release`.
- Onboarding the AI setup itself → see [docs/AI.md](docs/AI.md).
