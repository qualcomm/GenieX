# Contributing to geniex

Thanks for your interest in contributing! Whether you're fixing a typo, adding a backend, or landing a new feature, this guide takes you from `git clone` to a merged PR. The same rules apply to external and internal contributors — this is the single source of truth.

The commit and PR rules are **SemVer-driven**: they must encode enough information for [`/release`](.claude/commands/release.md) (and future automation) to derive the next version tag deterministically. The full tag procedure lives in [notes/release.md](notes/release.md).

## Getting started

Prerequisites:

- **[Bazelisk](https://github.com/bazelbuild/bazelisk)** — builds the Go CLI and fetches all CLI-side dependencies. The Bazel version is pinned in [.bazelversion](.bazelversion). Install: `winget install --id Bazel.Bazelisk` (Windows) or `bazelisk` from your package manager (Linux).
- **CMake** — builds the SDK bridge and native plugins via presets in [sdk/CMakePresets.json](sdk/CMakePresets.json).
- **Docker** — the Linux and Android SDK builds run inside prebuilt toolchain images (`qualcomm/geniex-toolchain-linux`, `qualcomm/geniex-toolchain-android`). A full Windows Snapdragon build additionally needs the Hexagon SDK, OpenCL SDK, and Windows Driver Kit.

Clone with submodules — the SDK vendors `third-party/llama.cpp` and `third-party/geniex-qairt`:

```bash
git clone --recursive https://github.com/qualcomm/GenieX.git
```

> [!IMPORTANT]
> **Build the SDK before the CLI.** In the default local-SDK mode the CLI links against `sdk/pkg-geniex/`, which must already exist. Full per-platform build commands (Windows ARM64, Linux, Android) live in [notes/build.md](notes/build.md); Python bindings in [bindings/python/README.md](bindings/python/README.md).

## Running tests

- **Go / CLI** — run via coverage rather than plain `test` (every package ships a placeholder test to stay in the coverage denominator):

  ```bash
  bazelisk coverage //... --combined_report=lcov
  ```

  The [`coverage`](.claude/skills/coverage/SKILL.md) skill wraps this and renders HTML.

- **SDK / plugins (Python)** — end-to-end pytest suite driven through the public binding. Model-free checks run anywhere; model and device cells need Snapdragon hardware (see [tests/README.md](tests/README.md)):

  ```bash
  pytest tests -m api                                  # model-free, runs anywhere
  pytest tests -m "api or (llama_cpp and device_cpu)"  # adds llama_cpp CPU cells
  GENIEX_DEVICE_TEST=1 pytest tests                    # full matrix, Snapdragon only
  ```

CI runs only the model-free shard on GitHub runners; the device matrix runs on real QDC hardware via [pr-check.yml](.github/workflows/pr-check.yml).

## Project structure

| Directory        | Contents                                                                       |
|------------------|--------------------------------------------------------------------------------|
| [sdk/](sdk/)     | Native C-ABI core (`include/geniex.h`), backend plugins (`llama_cpp`, `qairt`), Rust model-manager, benchmark. |
| [cli/](cli/)     | Go CLI (`cmd/`, `internal/`, `server/`, `release/`), built with Bazel.         |
| [bindings/](bindings/) | Language bindings: `android/`, `go/`, `python/`, `python-llama-cpp/`, `python-qairt/`. |
| [docs/](docs/)   | Documentation site (Mintlify).                                                 |
| [examples/](examples/) | Sample usage (`android/`, `python/`).                                    |
| [tests/](tests/) | SDK/plugin pytest suite + Go CLI black-box tests + QDC device harness.          |
| [scripts/](scripts/)   | Bazel helper rules.                                                      |
| [third-party/](third-party/) | Vendored `llama.cpp` and `geniex-qairt` submodules.               |
| [notes/](notes/) | Developer playbooks (build, run, release, bench, AI).                           |

## Making a change

### Branches

Format: `<type>/<short-topic>`. Base branch is `main`.

| Type      | Use for                                    | Example                     |
|-----------|--------------------------------------------|-----------------------------|
| `feat`    | New feature work.                          | `feat/ccache-sdk-build`     |
| `fix`     | Non-urgent bug fix targeting main.         | `fix/windows-dll-directory` |
| `hotfix`  | Urgent fix for a shipped release.          | `hotfix/signing-regression` |
| `chore`   | Tooling, deps, infra.                      | `chore/claude-framework`    |
| `docs`    | Documentation only.                        | `docs/release-procedure`    |
| `ci`      | CI config only.                            | `ci/add-windows-runner`     |
| `release` | Long-lived release-prep branches (rare).   | `release/0.5`               |

Personal dev branches like `perry/dev/<topic>` are allowed for shared WIP. Tag policy (which channels may be cut from which branches) lives in [notes/release.md](notes/release.md).

### Commits — Conventional Commits

```
<type>(<scope>)[!]: <subject>

<body>            # optional; only when the "why" is non-obvious
<footer>          # optional; BREAKING CHANGE, issue refs, Signed-off-by
```

Every commit MUST use one of these types. CI and `/release` derive the SemVer bump from the type.

| Type       | Meaning                                      | Bump               |
|------------|----------------------------------------------|--------------------|
| `feat`     | New user-visible feature.                    | MINOR              |
| `fix`      | Bug fix.                                     | PATCH              |
| `perf`     | Performance improvement, no behavior change. | PATCH              |
| `refactor` | Internal restructure, no behavior change.    | PATCH              |
| `docs`     | Documentation only.                          | PATCH              |
| `chore`    | Build, deps, tooling, misc.                  | PATCH              |
| `test`     | Test-only change.                            | PATCH              |
| `ci`       | CI config only.                              | PATCH              |
| `revert`   | Revert a prior commit.                       | no bump by default |

- **Scope** — one per commit, naming the area touched: `cli`, `sdk`, `python`, `android`, `go`, `bindings`, `server`, `build`, `release`, `ci`, `dx`, `docs`. Use `bindings` only when a change spans more than one language binding at once; a single-language change uses `go` / `python` / `android`. Introduce a new scope in the same PR that introduces the area.
- **Subject** — imperative mood (`add`, not `added`), ≤ 72 characters, no trailing period, describes *what* not *why*. Banned subjects (reviewers reject them — they make version derivation impossible): `update code`, `fix bug`, `misc`, `wip`, `tmp`, empty, placeholder, non-ASCII.
- **Body** — omit by default; add only when the "why" is non-obvious. Don't restate the diff. No co-authors, no emojis.
- **Breaking changes** — mark with `!` after the type/scope (`feat(cli)!: rename --model flag`) or a `BREAKING CHANGE:` footer. **Pre-1.0 rule:** the leading `X` in `vX.Y.Z` stays `0` until the first stable public release, so breaking changes bump **MINOR**, not MAJOR, while `X = 0`.

If local history is messy before opening a PR, tidy it with the [`reshape-pr-commits`](.claude/skills/reshape-pr-commits/SKILL.md) skill or `git rebase -i`.

### Sign your commits (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Every commit must carry a `Signed-off-by` line — add it with `-s`:

```bash
git commit -s -m "fix(sdk): guard null plugin handle"
```

The DCO check in CI rejects PRs whose commits are not signed off.

Keep each change focused — split independent changes into separate PRs.

## Code style & linting

Run the same checks CI runs before you commit. The authoritative list is [.github/workflows/_lint.yml](.github/workflows/_lint.yml); as of today:

| Language | Scope                                              | Command                                          |
|----------|----------------------------------------------------|--------------------------------------------------|
| C/C++    | touched files under `sdk/`, `cli/`, `bindings/python/` | `clang-format-18 -i <file>`                  |
| Python   | `bindings/python/**/*.py`                          | `ruff check --fix <file> && ruff format <file>`  |
| Go       | `cli/`                                             | `go mod tidy` (commit the `go.mod`/`go.sum` diff) |

When CI adds a check, this section needs no edit — the workflow is the source of truth.

### Changing public SDK headers

Public headers live under [sdk/include/](sdk/include/). Changing them requires updating every binding's FFI surface **in the same PR** — otherwise the binding crashes at load or first call.

| Binding        | FFI surface                                                                                                                    |
|----------------|-------------------------------------------------------------------------------------------------------------------------------|
| Python ctypes  | [bindings/python/geniex/_ffi/_api.py](bindings/python/geniex/_ffi/_api.py), [_types.py](bindings/python/geniex/_ffi/_types.py) |
| Go cgo         | [bindings/go/](bindings/go/)                                                                                                   |
| Android JNI    | [bindings/android/app/src/main/cpp/](bindings/android/app/src/main/cpp/)                                                       |

### Logging

Library code MUST route log output through the existing channels — no raw `std::cerr`, `printf`, Go `log`/`fmt.Println`, Python `print`, or direct `android.util.Log`.

| Area             | Use                                                          |
|------------------|--------------------------------------------------------------|
| SDK C/C++        | `GENIEX_LOG_TRACE/DEBUG/INFO/WARN/ERROR` macros              |
| Go (CLI+binding) | `log/slog`                                                   |
| Python           | `logging.getLogger("geniex")` (or a child logger)            |
| Android / JNI    | `android.util.Log` in Kotlin; JNI forwards through `geniex_set_log` |

The user-facing control is `GENIEX_LOG` (`trace`/`debug`/`info`/`warn`/`error`/`none`), read by each binding's logger.

## Opening a PR

- **Base**: `main`. **Merge strategy**: rebase.
- **Title**: MUST follow the Conventional Commits format above — reviewers and agents reject titles that don't.
- **Body**: `## Summary`, a `## Test plan` checklist, and a `Closes #<issue>` line. Follow the shape of recent merged PRs.
- **Checks**: the PR graph ([pr-check.yml](.github/workflows/pr-check.yml)) runs lint → build-sdk → CLI/Python/Android builds → docker. In parallel, [qcom-preflight-checks.yml](.github/workflows/qcom-preflight-checks.yml) runs Semgrep, DCO sign-off, copyright/license, commit-email, and dependency review. Resolve anything they flag before merge.
- **Reviewers**: no required reviewers today; request review from the owner of the area you touched — see [CODEOWNERS](CODEOWNERS).

## Reporting issues

Open an issue at [github.com/qualcomm/GenieX/issues](https://github.com/qualcomm/GenieX/issues). A good bug report includes a minimal repro, your environment (OS, device, backend, model), and relevant logs (`GENIEX_LOG=debug`). Feature requests are welcome as issues too — describe the use case, not just the implementation.

For **security vulnerabilities**, do **not** open a public issue — follow [SECURITY.md](SECURITY.md).

## Community & Code of Conduct

Participation is governed by our [Code of Conduct](CODE-OF-CONDUCT.md). Be respectful and constructive.

## Releases

See [notes/release.md](notes/release.md). The [`/release`](.claude/commands/release.md) command walks through the tag format, channel semantics (`alpha`/`beta`/`rc`/stable), and decision procedure.
