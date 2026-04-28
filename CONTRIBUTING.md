# Contributing to geniex

This document defines the **rules** contributors — human and AI — are
expected to follow when committing, branching, and opening pull
requests on this repository.

The rules are **SemVer-driven**: commit messages and PR titles must
encode enough information for [`/release`](.claude/commands/release.md)
(and any future automation) to derive the next version tag
**deterministically**, following the procedure in
[docs/release.md](docs/release.md).

## 1. Commits — Conventional Commits

Format:

```
<type>(<scope>)[!]: <subject>

<body>            # optional; only when the "why" is non-obvious
<footer>          # optional; BREAKING CHANGE, issue refs
```

### Types

Every commit MUST use one of these types. CI and `/release` derive the
SemVer bump from the type.

| Type       | Meaning                                     | Bump               |
|------------|---------------------------------------------|--------------------|
| `feat`     | New user-visible feature.                   | MINOR              |
| `fix`      | Bug fix.                                    | PATCH              |
| `perf`     | Performance improvement, no behavior change.| PATCH              |
| `refactor` | Internal restructure, no behavior change.   | PATCH              |
| `docs`     | Documentation only.                         | PATCH              |
| `chore`    | Build, deps, tooling, misc.                 | PATCH              |
| `test`     | Test-only change.                           | PATCH              |
| `ci`       | CI config only.                             | PATCH              |
| `revert`   | Revert a prior commit.                      | no bump by default |

### Breaking changes

Mark breaking changes with `!` after the type/scope, OR with a
`BREAKING CHANGE:` footer:

```
feat(cli)!: rename --model flag to --model-id
```

```
feat(sdk): restructure plugin loader

BREAKING CHANGE: plugins must now expose geniex_plugin_v2_init.
```

**Pre-1.0 rule.** This repo is still private and pre-1.0 — the leading
`X` in `vX.Y.Z` **must stay `0`** until the first public release.
Breaking changes bump **MINOR**, not MAJOR, while `X = 0`. See the
Versioning section of [docs/release.md](docs/release.md).

### Scope

One scope per commit. Use the area touched. Current scopes:

`cli`, `sdk`, `python`, `android`, `go`, `server`, `build`, `release`,
`ci`, `dx`, `docs`.

Introduce a new scope in the same PR that introduces the area it
covers.

### Subject

- imperative mood (`add`, not `added` / `adds`)
- ≤ 72 characters
- no trailing period
- describe *what*, not *why* (the body is for *why*)

**Banned subjects** — reviewers and agents must reject these; they
make version derivation impossible:

`update code`, `fix bug`, `misc`, `wip`, `tmp`, empty, placeholder
non-ASCII.

### Body

Omit by default. Add only when the "why" is non-obvious: a hidden
constraint, a subtle invariant, or a decision a future reader will
question. Do not restate the diff. No co-authors, no emojis, no
trailing summary lines.

### Shaping history

If local history is already messy, use the
[`reshape-pr-commits`](.claude/skills/reshape-pr-commits/SKILL.md)
skill rather than hand-rolling an interactive rebase.

## 2. Branches

Format: `<type>/<short-topic>`.

| Type       | Use for                                   | Example                       |
|------------|-------------------------------------------|-------------------------------|
| `feat`     | New feature work.                         | `feat/ccache-sdk-build`       |
| `fix`      | Non-urgent bug fix targeting main.        | `fix/windows-dll-directory`   |
| `hotfix`   | Urgent fix for a shipped release.         | `hotfix/signing-regression`   |
| `chore`    | Tooling, deps, infra.                     | `chore/claude-framework`      |
| `docs`     | Documentation only.                       | `docs/release-procedure`      |
| `ci`       | CI config only.                           | `ci/add-windows-runner`       |
| `release`  | Long-lived release-prep branches (rare). | `release/0.5`                 |

Base: `main`. Tag policy (which channels may be cut from which
branches) lives in [docs/release.md](docs/release.md).

### Personal dev branches

Personal branches like `perry/dev/<topic>` or `paul/dev/<topic>` are
allowed for shared WIP and may be merged directly.

Agents opening a PR from a personal branch MUST warn the user once
that the branch name does not follow the typed convention, and
proceed only after the user explicitly confirms. The PR title still
must follow the commit format in § 1.

## 3. Before you commit

Run the same checks CI runs. The authoritative list is
[.github/workflows/lint.yml](.github/workflows/lint.yml) — read it if
in doubt. As of today:

- **C/C++**: `clang-format` on touched files under `sdk/`, `cli/`,
  `bindings/python/`.
- **Python**: `ruff check` + `ruff format --check` on
  `bindings/python/**/*.py`.
- **Go**: `go mod tidy` clean inside `cli/`.

When CI adds a check, this section does not need editing — the link
is the source of truth.

Tests: see the section of [docs/build.md](docs/build.md) matching the
target you're touching.

## 4. Changing public SDK headers

Public SDK headers live under [sdk/include/](sdk/include/). Changing
them requires updating every binding's FFI surface **in the same
commit / PR** — otherwise the binding crashes at load or first call.

Surfaces to update:

- **Python ctypes** —
  [bindings/python/geniex/geniex_sdk/_api.py](bindings/python/geniex/geniex_sdk/_api.py),
  [_types.py](bindings/python/geniex/geniex_sdk/_types.py)
- **Go cgo** — [bindings/go/](bindings/go/)
- **Android JNI** — [bindings/android/app/src/main/cpp/](bindings/android/app/src/main/cpp/)

After updating one, ask whether the others need to move too.

## 5. Opening a PR

- **Base**: `main`.
- **Merge strategy**: squash merge. The final commit is the PR title,
  so **the PR title MUST follow the Conventional Commits format**
  defined in § 1. Reviewers and agents reject titles that do not.
- **Body**: use the shape you see in recent merged PRs —
  `## Summary`, optional subsections for large changes,
  `## Test plan` as a checklist, and a `Closes #<issue>` line.
- **Reviewers**: no required reviewers are configured today; request
  review from the owner of the area you are touching.

## 6. Releases

See [docs/release.md](docs/release.md). Tag format, channel semantics
(`alpha` / `beta` / `rc` / stable), and the decision procedure are
defined there. The [`/release`](.claude/commands/release.md) slash
command walks through it.
