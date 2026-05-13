# Release

Tag a commit with a `v`-prefixed [SemVer 2.0](https://semver.org/) string to trigger [`release.yml`](../.github/workflows/release.yml):

```bash
git tag v1.2.3 && git push origin v1.2.3
```

- Tags containing `-` are drafts (and push the sdist to TestPyPI).
- Bare `vX.Y.Z` tags publish immediately.
- Assets: `geniex-sdk-{linux,windows}-arm64-<tag>.zip`, `geniex-cli-linux-arm64-<tag>.tar.gz`, `geniex-cli-setup-windows-arm64-<tag>.exe`, `*.whl`, `*.aar`, and per-file `.sha256` sidecars.
- Re-running the same tag via **Actions → Release → Run workflow** is safe — assets are replaced.
- S3 mirror at `s3://qaihub-public-assets/qai-hub-geniex/`: the Windows SDK zip and the `geniex-cli-setup-windows-arm64-<tag>.exe` installer go up on every tag; the mutable pointer `geniex-cli.exe` is only advanced on stable (non-prerelease) tags and is served with `Cache-Control: no-cache` so downloaders always pick up the newest release.

> For the developer/user side of HTP signing (cert import, test-signing), see [run.md § Self-signed fallback](run.md#self-signed-fallback).

## Versioning

Tags follow [SemVer 2.0](https://semver.org/) with a `v` prefix: `vX.Y.Z` for stable, `vX.Y.Z-<channel>.<n>` for pre-releases.

### Digits

| Bump        | Meaning                                                       | Example triggers                                                                                |
|-------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| MAJOR (`X`) | Breaking change to any public surface. Consumers must adapt.  | CLI flag removed/renamed, SDK header signature changed, Python API removed, config key renamed. |
| MINOR (`Y`) | Backwards-compatible feature addition.                        | New backend, new model support, new CLI subcommand, new SDK function.                           |
| PATCH (`Z`) | Backwards-compatible fix or cleanup.                          | Bug fix, dependency bump, doc/CI-only change, internal refactor.                                |

**Pre-1.0 rule.** The project is still pre-1.0 (`X = 0`). Do **not** bump MAJOR while private/unreleased — keep `X = 0`. Breaking changes bump **MINOR** (`0.Y → 0.(Y+1)`, resetting `Z` to 0) and must be flagged in the release notes. The project graduates to `X = 1` only on first public release.

### Channels

Pre-release channels communicate how ready a build is. Ordering: `alpha < beta < rc < stable`.

| Channel   | Purpose                                                        | Allowed branch         | Still allowed to change       |
|-----------|----------------------------------------------------------------|------------------------|-------------------------------|
| `alpha.n` | Share in-progress builds; feature shape may still move.        | feature branch or main | Anything, including breaking. |
| `beta.n`  | Feature-complete for the target `X.Y.Z`; seeking feedback.     | `main`                 | Bug fixes and polish only.    |
| `rc.n`    | Release candidate — "will ship unless we find a bug".          | `main`                 | Bug fixes only.               |
| stable    | Published release.                                             | `main`                 | Nothing — cut a new bump.     |

Rules:

- **Always pass through at least one `-rc.n` on `main` before stable.**
- **A stable tag must be on a commit whose HTP bundle is Microsoft-signed**, not self-signed. See [Hexagon HTP signing](#hexagon-htp-signing); if unsure, run `gh run list --workflow release.yml --limit 5` and confirm the SDK artifact name does **not** end in `-selfsigned`. If it does, finish the signing promotion first.
- **Never re-use or move a published tag.** Retract via a new patch.
- **`alpha.n` tags on feature branches are disposable** — don't rebase the tagged commit afterward.

### Decision procedure

This is the algorithm `/release` follows. Apply in order.

1. **Find the latest stable tag** `v0.A.B` locally — avoids a `gh` round-trip:

   ```bash
   git tag --sort=-v:refname --list "v[0-9]*.[0-9]*.[0-9]*" | grep -v -- '-' | head -1
   ```

   Don't use `git describe` — it only looks at HEAD's ancestor chain and will miss stable tags cut on side branches. If the command prints nothing, target `v0.1.0` and skip to step 3.
2. **Pick the target `X.Y.Z`** from `git log v0.A.B..HEAD --format="%h %s" --stat` (subject + touched files in one pass; run `git show <sha>` only if that's still ambiguous):
   - any commit with a breaking change → `v0.(A+1).0` (while `X = 0`, breaking bumps MINOR);
   - else any feature commit → `v0.(A+1).0`;
   - else → `v0.A.(B+1)`.

   Read both subjects and diffs when the subject is ambiguous — Conventional-Commits prefixes (`feat:`, `fix:`, `feat!:`) are a hint, not a contract; the repo does not enforce them.
3. **Pick the channel** by cycle position for that target:
   - first tag toward a new target, on a feature branch → `alpha.1`;
   - first tag toward a new target, on `main` → `rc.1` (skip beta unless explicitly requested; most cycles go straight to rc);
   - already in cycle → advance within the channel (`-rc.1` → `-rc.2`) or move forward one channel (`-beta.3` → `-rc.1`, resetting `n`). Channels only move forward on a given `X.Y.Z`;
   - all `-rc.n` green + HTP Microsoft-signed → cut bare `vX.Y.Z`.
4. **If a breaking change lands mid-cycle** and raises the target: abandon the current `X.Y.Z` (leave existing tags in place — don't retract) and restart at `v0.(new)-alpha.1` or `-rc.1`.
5. **Counter `n`** resets per `X.Y.Z` and per channel: `0.4.0-alpha.{1,2}` → `0.4.0-beta.{1}` → `0.4.0-rc.{1,2}` → `0.4.0`.

### Worked examples

- Latest stable `v0.3.2`. `git log v0.3.2..HEAD` has one `fix:` and one `docs:`. On `main`, first tag → `v0.3.3-rc.1`, then `v0.3.3`.
- Latest stable `v0.3.2`. Log contains a `feat:` adding a new backend. On a feature branch → `v0.4.0-alpha.1`; after merge to `main` → `v0.4.0-rc.1` → `v0.4.0`.
- On `v0.4.0-rc.2`, a CLI flag rename (breaking) lands. Target rises from `0.4.0` to `0.5.0`. Leave `-rc.2` alone; next tag is `v0.5.0-alpha.1` or `v0.5.0-rc.1` depending on the branch.

## Hexagon HTP signing

The Windows ARM64 SDK ships `libggml-htp.cat` plus `libggml-htp-v{68,69,73,75,79,81}.so` — Windows refuses to load them unsigned. Release CI runs an `overlay-htp` job **before** `build-cli` that `curl`s `s3://qaihub-public-assets/llama-cpp/libggml-htp-<sha>.zip`, where `<sha>` is the `third-party/llama.cpp` short SHA. Both the installer and the SDK zip end up with the same HTP files:

- **Hit** — overlay the Microsoft-signed files into the SDK artifact; `build-cli` packages them into the installer; release normally.
- **Miss** — keep the self-signed build. The SDK name gets a `-selfsigned` suffix, and the release also carries `ggml-htp-v1.cer` (users import it) and `libggml-htp-to-sign-<sha>.zip` (operators submit it for signing).

The S3 bundle must contain exactly these eight files at the zip root: `libggml-htp.cat`, `libggml-htp.inf`, and `libggml-htp-v{68,69,73,75,79,81}.so`.

### Promoting self-signed → Microsoft-signed

1. Download `libggml-htp-to-sign-<sha>.zip` from the draft release.
2. Submit for Microsoft signing.
3. Upload the result to `s3://qaihub-public-assets/llama-cpp/libggml-htp-<sha>.zip`.
4. Re-run the Release workflow for the same tag.
