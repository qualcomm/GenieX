# Release

Push a SemVer 2.0 tag prefixed with `v` to trigger [`release.yml`](../.github/workflows/release.yml):

```bash
git tag v1.2.3 && git push origin v1.2.3
```

- Any pre-release tag (anything with a `-<suffix>`) → release is created as
  **draft** (e.g. `v1.2.3-alpha.1`, `v1.2.3-rc.1`).
- Stable tags (`v1.2.3` with no suffix) → published immediately.
- Pre-release tags matching `v*-alpha.*` / `v*-beta.*` / `v*-rc.*` also push
  the sdist to TestPyPI; stable tags skip that.
- Release body is auto-generated from commit history (`generate_release_notes`).
- Assets: `geniex-{sdk,cli}-{linux,windows}-arm64-<tag>.zip`, `*.whl`, per-file
  `.sha256` sidecars.
- Verify a file: `sha256sum -c <file>.sha256`.

Manual re-run: **Actions → Release → Run workflow** with the tag name. The
release.js script is idempotent (duplicates are replaced), so the same tag can
be re-run safely.

## Versioning policy

We follow [SemVer 2.0](https://semver.org/). The leading `v` is the trigger
prefix, everything after it is pure SemVer.

### When to bump MAJOR / MINOR / PATCH

| Bump   | Trigger                                                                           | Example             |
|--------|-----------------------------------------------------------------------------------|---------------------|
| MAJOR  | Breaking change to a public surface (CLI flags, SDK headers, Python API, config). | `v0.4.2 → v1.0.0`   |
| MINOR  | New feature, new model/backend support, non-breaking SDK additions.               | `v1.2.0 → v1.3.0`   |
| PATCH  | Bug fix, dependency bump, doc/CI change with no behavior impact.                  | `v1.3.0 → v1.3.1`   |

While we're on `0.y.z` (pre-1.0), MINOR is allowed to carry breaking changes —
track them loudly in the release notes. Graduate to `1.0.0` once the public
CLI/SDK surface is committed.

### Pre-release channels

Pre-releases use a dotted suffix: `v<X.Y.Z>-<channel>.<n>`. All are cut as
drafts so we can sanity-check assets before hitting publish.

| Suffix            | When to use                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| `-alpha.<n>`      | Early internal drop. Shape of the change may still move; no promises.       |
| `-beta.<n>`       | Feature frozen, looking for broader internal/external feedback.             |
| `-rc.<n>`         | Release candidate. Only bug fixes between `-rc.n` and the stable tag.       |

Counters restart per `X.Y.Z`: `v1.3.0-alpha.1 → -alpha.2 → -rc.1 → -rc.2 → v1.3.0`.
Don't skip from `-alpha` to stable without an `-rc` — that's the gate where we
validate the actual release assets.

### Cutting a release

1. Land all target PRs on `master`.
2. Tag the commit: `git tag vX.Y.Z-rc.1 && git push origin vX.Y.Z-rc.1`.
3. Verify the draft release (assets, SHA256, HTP signing flavor — see below).
4. If good, tag stable: `git tag vX.Y.Z && git push origin vX.Y.Z`. If issues,
   fix on `master` and cut `-rc.2`.

Never move a tag once published. To retract, create a new patch release.

## Hexagon HTP signing

The Windows-ARM64 SDK ships `libggml-htp.cat` + six `libggml-htp-v*.so` files
that must be signed before Windows will load them. Release CI picks the signing
source based on whether a Microsoft-signed bundle exists on S3 for the current
`third-party/llama.cpp` submodule commit.

```
                                         ┌────────────────────────────┐
llama.cpp short SHA ──▶  curl s3://…/llama-cpp/libggml-htp-<sha>.zip ─┤
                                         └────────────────────────────┘
                            hit (200)                    miss (4xx)
                               │                             │
                               ▼                             ▼
              overlay signed files into         keep self-signed build
              sdk-pkg/lib/llama_cpp/            and ship:
              → SDK published as                → SDK suffixed -selfsigned
                geniex-sdk-…-<tag>.zip          → ggml-htp-v1.cer
                                                → libggml-htp-to-sign-<sha>.zip
```

- **S3 bucket**: `s3://qaihub-public-assets/llama-cpp/libggml-htp-<6char>.zip`
- Signed bundle layout (required): `libggml-htp.cat`, `libggml-htp.inf`,
  `libggml-htp-v{68,69,73,75,79,81}.so`.

### Operator runbook (bumping llama.cpp / landing a new signed bundle)

When the `third-party/llama.cpp` submodule points to a new SHA, Release CI will
fall back to the self-signed flavor until the matching signed bundle lands on
S3. To promote a release from self-signed to Microsoft-signed:

1. Cut a release with the new llama.cpp SHA. The draft will include
   `libggml-htp-to-sign-<sha>.zip` — download it.
2. Submit that zip for Microsoft signing.
3. Upload the signed result to
   `s3://qaihub-public-assets/llama-cpp/libggml-htp-<sha>.zip`. File layout
   must match the "signed bundle layout" above.
4. Re-run the Release workflow for the same tag (**Actions → Release → Run
   workflow**). CI re-publishes without the `-selfsigned` suffix / `.cer`, and
   the release body updates to the Microsoft-signed note.
