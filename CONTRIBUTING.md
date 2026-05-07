# Contributing to geniex

Rules contributors — human and AI — follow when committing, branching, and opening pull requests.

The rules are **SemVer-driven**: commit messages and PR titles must encode enough information for [`/release`](.claude/commands/release.md) (and future automation) to derive the next version tag deterministically. The full tag procedure lives in [notes/release.md](notes/release.md).

## 1. Commits — Conventional Commits

```
<type>(<scope>)[!]: <subject>

<body>            # optional; only when the "why" is non-obvious
<footer>          # optional; BREAKING CHANGE, issue refs
```

### Types

Every commit MUST use one of these. CI and `/release` derive the SemVer bump from the type.

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

### Breaking changes

Mark with `!` after the type/scope, or with a `BREAKING CHANGE:` footer:

```
feat(cli)!: rename --model flag to --model-id
```

```
feat(sdk): restructure plugin loader

BREAKING CHANGE: plugins must now expose geniex_plugin_v2_init.
```

**Pre-1.0 rule.** The repo is private and pre-1.0 — the leading `X` in `vX.Y.Z` **must stay `0`** until the first public release. Breaking changes bump **MINOR**, not MAJOR, while `X = 0`.

### Scope

One scope per commit, naming the area touched. Current scopes: `cli`, `sdk`, `python`, `android`, `go`, `server`, `build`, `release`, `ci`, `dx`, `docs`. Introduce a new scope in the same PR that introduces the area.

### Subject

- Imperative mood (`add`, not `added` / `adds`).
- ≤ 72 characters.
- No trailing period.
- Describe *what*, not *why* — the body is for *why*.

**Banned subjects** — reviewers and agents must reject these; they make version derivation impossible: `update code`, `fix bug`, `misc`, `wip`, `tmp`, empty, placeholder, non-ASCII.

### Body

Omit by default. Add only when the "why" is non-obvious: a hidden constraint, a subtle invariant, or a decision a future reader will question. Do not restate the diff. No co-authors, no emojis, no trailing summary lines.

### Shaping history

If local history is messy before opening a PR, prefer the [`reshape-pr-commits`](.claude/skills/reshape-pr-commits/SKILL.md) skill. Without Claude Code, use `git rebase -i` to squash/fixup/reword the equivalent commits manually.

## 2. Branches

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

Tag policy (which channels may be cut from which branches) lives in [notes/release.md](notes/release.md).

### Personal dev branches

Personal branches like `perry/dev/<topic>` or `paul/dev/<topic>` are allowed for shared WIP and may be merged directly.

Agents opening a PR from a personal branch MUST warn the user once that the branch name does not follow the typed convention, and proceed only after explicit confirmation. The PR title still must follow the commit format in § 1.

## 3. Before you commit

Run the same checks CI runs — the authoritative list is [.github/workflows/lint.yml](.github/workflows/lint.yml). As of today:

- **C/C++**: `clang-format` on touched files under `sdk/`, `cli/`, `bindings/python/`.
- **Python**: `ruff check` + `ruff format --check` on `bindings/python/**/*.py`.
- **Go**: `go mod tidy` clean inside `cli/`.

When CI adds a check, this section needs no edit — the workflow is the source of truth.

Tests: see the section of [notes/build.md](notes/build.md) matching your target.

## 4. Changing public SDK headers

Public headers live under [sdk/include/](sdk/include/). Changing them requires updating every binding's FFI surface **in the same commit / PR** — otherwise the binding crashes at load or first call.

| Binding        | FFI surface                                                                                                                 |
|----------------|-----------------------------------------------------------------------------------------------------------------------------|
| Python ctypes  | [bindings/python/geniex/_ffi/_api.py](bindings/python/geniex/_ffi/_api.py), [_types.py](bindings/python/geniex/_ffi/_types.py) |
| Go cgo         | [bindings/go/](bindings/go/)                                                                                                |
| Android JNI    | [bindings/android/app/src/main/cpp/](bindings/android/app/src/main/cpp/)                                                    |

After updating one binding, ask whether the others need to move too.

### Logging

New code MUST route log output through the existing channels — no raw
`std::cerr`, `printf`, Go `log` / `fmt.Println`, Python `print`, or direct
`android.util.Log` calls inside library code paths.

| Area             | Use                                                |
|------------------|----------------------------------------------------|
| SDK C/C++        | `GENIEX_LOG_TRACE/DEBUG/INFO/WARN/ERROR` macros    |
| Go (CLI+binding) | `log/slog`                                         |
| Python           | `logging.getLogger("geniex")` (or a child logger)  |
| Android / JNI    | `android.util.Log` in Kotlin; JNI forwards through `geniex_set_log` |

The user-facing control is `GENIEX_LOG` (`trace`/`debug`/`info`/`warn`/`error`/`none`),
with bindings exposing an API override (`geniex.set_log_level` / `geniex_sdk.SetLogLevel` /
`GeniexSdk.setLogLevel`).

## 5. Opening a PR

- **Base**: `main`.
- **Merge strategy**: squash merge. The final commit is the PR title, so **the PR title MUST follow the Conventional Commits format** in § 1. Reviewers and agents reject titles that do not.
- **Body**: follow the shape seen in recent merged PRs — `## Summary`, optional subsections for large changes, `## Test plan` as a checklist, and a `Closes #<issue>` line.
- **Reviewers**: no required reviewers today; request review from the owner of the area you touched.

## 6. Releases

See [notes/release.md](notes/release.md). The [`/release`](.claude/commands/release.md) slash command walks through the tag format, channel semantics (`alpha` / `beta` / `rc` / stable), and decision procedure.
