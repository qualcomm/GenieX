# Release Process

GenieX releases are driven by the [`release.yml`](../.github/workflows/release.yml)
workflow, which is triggered automatically when a tag matching `v*` is pushed,
or can be dispatched manually from the Actions tab.

## Tag convention (SemVer 2.0)

Tags **must** follow [Semantic Versioning 2.0](https://semver.org/) with a
required `v` prefix:

```
v<MAJOR>.<MINOR>.<PATCH>[-<pre-release>][+<build-metadata>]
```

Examples that pass validation:

| Tag                          | GitHub Release flag | PEP 440 wheel version |
|------------------------------|---------------------|-----------------------|
| `v1.0.0`                     | regular             | `1.0.0`               |
| `v1.2.3`                     | regular             | `1.2.3`               |
| `v1.0.0-alpha.1`             | pre-release         | `1.0.0a1`             |
| `v1.0.0-beta.2`              | pre-release         | `1.0.0b2`             |
| `v1.0.0-rc.1`                | pre-release         | `1.0.0rc1`            |
| `v1.0.0+build.5`             | regular             | `1.0.0`               |
| `v1.0.0-alpha.1+exp.sha.abc` | pre-release         | `1.0.0a1`             |

Anything else (e.g. `1.0.0`, `v1.0`, `v01.0.0`) is rejected by the
`Validate release tag (SemVer 2.0)` step.

Tags that contain a `-` pre-release identifier are automatically published as
**Pre-release** on GitHub.

## Creating a release

### Via git tag (normal path)

```bash
git checkout master
git pull --ff-only
git tag v1.2.3
git push origin v1.2.3
```

### Via workflow_dispatch (manual path)

Useful for re-running a release if artifact upload failed, or for testing on a
branch. In the GitHub UI go to **Actions → Release → Run workflow** and supply
the tag to release (e.g. `v1.2.3`). The tag must already exist on the repo.

## What the workflow produces

The `publish-github-release` job uploads the following assets to the release:

- `geniex-sdk-linux-arm64-<tag>.zip`
- `geniex-sdk-windows-arm64-<tag>.zip`
- `geniex-cli-linux-arm64-<tag>.zip`
- `geniex-cli-windows-arm64-<tag>.zip`
- `geniex-*.whl` (Linux ARM64 + Windows ARM64 wheels)
- `SHA256SUMS-<tag>.txt` — SHA-256 checksums of every archive/wheel above

Verify downloads with:

```bash
sha256sum -c SHA256SUMS-v1.2.3.txt
```

## Release notes

The workflow generates a `RELEASE_NOTES.md` body from commits between the
previous tag and the current tag, grouped by conventional-commit prefix
(`feat:`, `fix:`, `perf:`, `docs:`, `build:`, `ci:`, `refactor:`, `chore:`).
Commits that don't match a known prefix are placed under **Other**.

For the very first release on a fresh repo (no previous tag), the body is
marked as _Initial release_ and contains every commit reachable from the tag.
