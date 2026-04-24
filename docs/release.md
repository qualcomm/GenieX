# Release

Tag a commit with a `v`-prefixed [SemVer 2.0](https://semver.org/) string to
trigger [`release.yml`](../.github/workflows/release.yml):

```bash
git tag v1.2.3 && git push origin v1.2.3
```

Tags containing `-` are drafts (and push the sdist to TestPyPI); bare
`vX.Y.Z` tags publish immediately. Assets:
`geniex-{sdk,cli}-{linux,windows}-arm64-<tag>.zip`, `*.whl`, and per-file
`.sha256` sidecars. Re-running the same tag (**Actions → Release → Run
workflow**) is safe — assets are replaced.

## Versioning

Tags are [SemVer 2.0](https://semver.org/) with a `v` prefix: `vX.Y.Z`
for stable, `vX.Y.Z-<channel>.<n>` for pre-releases.

### What each digit means

| Bump  | Meaning                                                            | Example triggers                                                                                   |
|-------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| MAJOR (`X`) | Breaking change to any public surface. Consumers must adapt.  | CLI flag removed/renamed, SDK header signature changed, Python API removed, config key renamed.    |
| MINOR (`Y`) | Backwards-compatible feature addition.                        | New backend, new model support, new CLI subcommand, new SDK function (existing ones unchanged).    |
| PATCH (`Z`) | Backwards-compatible fix or cleanup.                          | Bug fix, dependency bump, doc/CI-only change, internal refactor.                                   |

**Project is still pre-1.0 (`X = 0`).** Do **not** bump MAJOR while the
project is private / unreleased — keep `X = 0`. Breaking changes bump
**MINOR** (`0.Y → 0.(Y+1)`, resetting `Z` to 0) and must be flagged in
the release notes. The project will graduate to `X = 1` only when it is
publicly released and ready to commit to backwards compatibility.

### What each channel means

Pre-release channels communicate **how ready a build is**. Ordering:
`alpha < beta < rc < stable`.

| Channel  | Audience / purpose                                             | Allowed branch        | Still allowed to change          |
|----------|----------------------------------------------------------------|-----------------------|----------------------------------|
| `alpha.n` | Share in-progress builds. Shape of features may still move.   | feature branch or master | Anything, including breaking.  |
| `beta.n`  | Feature-complete for the target `X.Y.Z`; seeking feedback.    | `master`              | Bug fixes and polish only.       |
| `rc.n`    | Release candidate. Treat as "will ship unless we find a bug". | `master`              | Bug fixes only.                  |
| stable    | Published release.                                             | `master`              | Nothing — cut a new PATCH/MINOR. |

Rules:

- **Always pass through at least one `-rc.n` on `master` before stable.**
- **A stable tag must be on a commit whose HTP bundle is Microsoft-signed** —
  not self-signed. See [Hexagon HTP signing](#hexagon-htp-signing); if
  unsure, check the latest release run (`gh run list --workflow release.yml --limit 5`)
  and confirm the SDK artifact name does **not** end in `-selfsigned`. If
  it does, ask the user to complete the signing promotion first.
- **Don't re-use or move published tags.** Retract via a new patch.
- **`alpha.n` tags on feature branches are disposable** — don't rebase
  the tagged commit afterward.

### Decision procedure (how to pick the next tag)

This is the algorithm `/release` follows. Apply it in order.

1. **Find the latest stable tag** `v0.A.B` — `gh release list --limit 20`
   (filter out pre-releases). If none exists, the target is `v0.1.0`
   and skip to step 3.
2. **Decide the target `X.Y.Z`** by scanning `git log v0.A.B..HEAD`:
   - any commit introducing a breaking change → target is `v0.(A+1).0`
     (while `X = 0`, breaking bumps MINOR, not MAJOR)
   - else any commit adding a feature → target is `v0.(A+1).0`
   - else → target is `v0.A.(B+1)`

   Look at both the commit subjects and the diff when the subject is
   ambiguous — Conventional-Commits-style prefixes (`feat:`, `fix:`,
   `feat!:`) are a hint, not a contract; the repo does not enforce them.
3. **Pick the channel** based on where you are in the cycle for that
   target `X.Y.Z`:
   - First tag toward a new target, on a feature branch → `alpha.1`
   - First tag toward a new target, on `master`       → `rc.1`
     (skip beta unless the user asks for one; most cycles go straight
     to rc)
   - Already in cycle → advance within the current channel (`-rc.1`
     → `-rc.2`) or move forward one channel (`-beta.3` → `-rc.1`,
     resetting `n`). **Channels only move forward on a given `X.Y.Z`.**
   - All `-rc.n` green + HTP signed → cut bare `vX.Y.Z`.
4. **If a breaking change lands mid-cycle** that raises the target,
   abandon the current `X.Y.Z` (leave existing tags in place — don't
   retract) and restart at `v0.(new)-alpha.1` / `-rc.1`.
5. **Counter `n`** resets per `X.Y.Z` **and** per channel:
   `0.4.0-alpha.{1,2}` → `0.4.0-beta.{1}` → `0.4.0-rc.{1,2}` →
   `0.4.0`.

#### Worked examples

- Latest stable is `v0.3.2`. `git log v0.3.2..HEAD` contains one `fix:`
  and one `docs:`. On `master`, first tag → `v0.3.3-rc.1`. After QA →
  `v0.3.3`.
- Latest stable is `v0.3.2`. Log contains a `feat:` adding a new
  backend. On a feature branch → `v0.4.0-alpha.1`. After merge to
  `master` → `v0.4.0-rc.1` → `v0.4.0`.
- On `v0.4.0-rc.2`, a CLI flag rename (breaking) lands. Target rises
  from `0.4.0` to `0.5.0`. Leave `-rc.2` alone; next tag is
  `v0.5.0-alpha.1` or `v0.5.0-rc.1` depending on the branch.

## Hexagon HTP signing

The Windows-ARM64 SDK ships `libggml-htp.cat` + `libggml-htp-v{68,69,73,75,79,81}.so`,
which Windows won't load unsigned. Release CI `curl`s
`s3://qaihub-public-assets/llama-cpp/libggml-htp-<sha>.zip` (where `<sha>` is
the `third-party/llama.cpp` short SHA) in an `overlay-htp` job that runs
**before** `build-cli`, so both the installer and the SDK zip contain the same
set of HTP files:

- **Hit** → overlay the signed files into the SDK artifact; `build-cli` packages
  them into the installer; release normally.
- **Miss** → keep the self-signed build; SDK name gets a `-selfsigned` suffix,
  and the release also carries `ggml-htp-v1.cer` (users import) plus
  `libggml-htp-to-sign-<sha>.zip` (operators submit for signing).

The S3 bundle must contain exactly: `libggml-htp.cat`, `libggml-htp.inf`, and
the six `libggml-htp-v*.so` files at the zip root.

### Promoting self-signed → Microsoft-signed

1. Download `libggml-htp-to-sign-<sha>.zip` from the draft release.
2. Submit for Microsoft signing.
3. Upload the result to `s3://qaihub-public-assets/llama-cpp/libggml-htp-<sha>.zip`.
4. Re-run the Release workflow for the same tag.
