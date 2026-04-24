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

| Bump  | Trigger                                                  |
|-------|----------------------------------------------------------|
| MAJOR | Breaking change to CLI flags, SDK headers, Python API, config. |
| MINOR | New feature or backend/model support.                    |
| PATCH | Bug fix, dep bump, doc/CI change.                        |

While on `0.y.z`, MINOR may carry breaking changes — flag them in the notes.

Pre-releases use `vX.Y.Z-<channel>.<n>`, counter resets per `X.Y.Z`:

- `alpha.n` — shape may still move. May be tagged on a feature branch to share
  test builds before merge; treat these tags as disposable and don't rebase
  the underlying commit after tagging.
- `beta.n`  — feature frozen, seeking feedback. Tag on `master`.
- `rc.n`    — release candidate. Tag on `master`. For internal pre-ship
  verification only; only bug fixes between `rc.n` and stable.

Always pass through at least one `-rc` on `master` before stable. Don't re-use
or move published tags — retract via a new patch.

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
