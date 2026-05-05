# Hexagon HTP Signing Certificate (fallback)

These are self-signed test certificates used to sign Hexagon HTP ops libraries
(`libggml-htp-*.so` → `libggml-htp.cat`) during CI builds. They are only
shipped to end users when Release CI cannot find a Microsoft-signed bundle for
the current `third-party/llama.cpp` SHA on S3 — see
[notes/release.md](../../../notes/release.md#hexagon-htp-signing). When the
signed bundle is available, Release CI overlays it and drops these certs from
the release assets.

| File | Purpose |
|------|---------|
| `ggml-htp-v1.pfx` | Private key + cert bundle consumed by the build (see `HEXAGON_HTP_CERT`) |
| `ggml-htp-v1.cer` | Public certificate shipped to end users (import into `Trusted Root Certification Authorities` and `Trusted Publishers`) |
| `ggml-htp-v1.pvk` | Raw private key, kept for re-signing |

## Why checked in

Temporary. Unblocks CI end-to-end. The pfx/pvk will be moved into GitHub Actions
secrets in a follow-up; the cer will remain distributed alongside releases.

## End-user install (Windows on Snapdragon)

```powershell
bcdedit /set TESTSIGNING ON            # requires reboot, may need Secure Boot off
# Import ggml-htp-v1.cer into both stores via certlm.msc:
#   - Trusted Root Certification Authorities
#   - Trusted Publishers
```

See `third-party/llama.cpp/docs/backend/snapdragon/windows.md` for background.
