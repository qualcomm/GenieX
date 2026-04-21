# Release

Push a SemVer 2.0 tag prefixed with `v` to trigger [`release.yml`](../.github/workflows/release.yml):

```bash
git tag v1.2.3 && git push origin v1.2.3
```

- `-rc` in the tag → release is created as **draft** (e.g. `v1.2.3-rc.1`).
- Release body is auto-generated from commit history (`generate_release_notes`).
- Assets: `geniex-{sdk,cli}-{linux,windows}-arm64-<tag>.zip`, `*.whl`, `SHA256SUMS-<tag>.txt`.
- Verify: `sha256sum -c SHA256SUMS-<tag>.txt`.

Manual re-run: **Actions → Release → Run workflow** with the tag name.
