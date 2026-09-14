# CLAUDE.md

## Project

Multi-platform AI inference runtime (Snapdragon / Hexagon focus).
Languages: C/C++ (SDK), Go (CLI), Python (bindings), Java/JNI (Android).
Build systems: Bazel (CLI) + CMake (SDK).

## Hard constraints

- **Never move or reuse a published git tag.** If the wrong tag shipped, cut a higher one.
- Do not modify third-party code.
- **Follow [CONTRIBUTING.md](CONTRIBUTING.md)** for branch naming, commit / PR title format, pre-commit checks, and the FFI-update rule when changing public SDK headers.

## Workflows

- Build anything (CLI / SDK bridge / release installer) → run `/build`.
- Cut a release / bump the version → run `/release`.
- Onboarding the AI setup itself → see [notes/AI.md](notes/AI.md).
