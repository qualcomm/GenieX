---
name: build
description: Build the GenieX SDK, CLI, Python bindings, or release packages, and run Go test coverage. Use when the user asks to build, rebuild, package, or measure coverage.
argument-hint: "[sdk | cli | python | installer | coverage] [platform]"
---

# Build

[notes/build.md](../../../notes/build.md) is the source of truth. Read the section for the target and platform, then run it. Don't restate or improvise commands. If the notes are wrong, fix them in the same change.

- **Order**: the SDK first (CMake → `sdk/pkg-geniex/`), then the CLI (Bazel links against it). If `sdk/pkg-geniex/lib/` lacks the bridge for this platform, build the SDK before the CLI.
- **Rebuild the SDK** after any change under `sdk/` or `third-party/`. CLI-only changes need only Bazel.
- **Windows**: check the prerequisites in notes/build.md first (symlink rights, long paths, short Hexagon toolchain path). Most "mysterious" failures are these.
- **Linux / Android**: build inside the toolchain container named in notes/build.md, not on the host.
- **Coverage**: `bazelisk coverage //... --combined_report=lcov`, then render with `genhtml` (lcov 2.x renders cleanly; 1.x ignores exclusion markers).
- After building, verify the binary you run is the one you built (see the `geniex` skill: an installed CLI on PATH can shadow it).
