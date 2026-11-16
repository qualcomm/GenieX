# CLAUDE.md

## Project

On-device AI inference runtime for Snapdragon (CPU / Adreno GPU / Hexagon NPU).
Languages: C/C++ (SDK, plugins), Rust (model manager), Go (CLI), Python (bindings), Kotlin/Java (Android).
Build: CMake (SDK) → Bazel (CLI); Gradle for Android.

## Hard constraints

- Never move or reuse a published git tag. If the wrong tag shipped, cut a higher one.
- Do not modify code under `third-party/`; patch it via `sdk/patches/`.
- Follow [CONTRIBUTING.md](CONTRIBUTING.md) for branches, commits (signed off), PR titles, and the FFI-update rule.
- `qualcomm/GenieX` is public: no credentials, internal hostnames, or customer names in code, docs, or PR text.

## Where things are

| Need | Go to |
|---|---|
| Build / package / coverage | `/build` → [notes/build.md](notes/build.md) |
| Run the CLI, flags, debugging | `geniex` skill → [docs/en/run/cli/reference.mdx](docs/en/run/cli/reference.mdx) |
| Release a tag | `/release` (manual only) → [notes/release.md](notes/release.md) |
| Runtime internals (compute units, power mode, QAIRT, HTP signing) | [notes/run.md](notes/run.md) |
| QDC benchmark / accuracy CI | [notes/bench.md](notes/bench.md) |
