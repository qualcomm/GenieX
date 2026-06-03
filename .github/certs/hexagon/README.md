# Hexagon HTP Signing Certificate (fallback)

This is the self-signed test certificate used to sign Hexagon HTP ops libraries
(`libggml-htp-*.so` → `libggml-htp.cat`) during CI builds. It is only
shipped to end users when Release CI cannot find a Microsoft-signed bundle for
the current `third-party/llama.cpp` SHA on S3 — see
[notes/release.md](../../../notes/release.md#hexagon-htp-signing). When the
signed bundle is available, Release CI overlays it and drops the cert from
the release assets.

| File | Purpose |
|------|---------|
| `ggml-htp-v1.cer` | Public certificate shipped to end users (import into `Trusted Root Certification Authorities` and `Trusted Publishers`) |

## Private key

The signing private key is **not** checked in. It lives in the
`HEXAGON_HTP_CERT_PFX` GitHub Actions secret (base64 of the `.pfx`), which the
`Configure HTP signing cert` step in `_build-sdk.yml` decodes to `HEXAGON_HTP_CERT`
at build time. To rotate or re-sign locally, regenerate the cert/key with the
`makecert` / `pvk2pfx` commands in
[notes/run.md § Self-signed fallback](../../../notes/run.md#self-signed-fallback)
and `gh secret set HEXAGON_HTP_CERT_PFX < <(base64 -w0 ggml-htp-v1.pfx)`.

## End-user install (Windows on Snapdragon)

```powershell
bcdedit /set TESTSIGNING ON            # requires reboot, may need Secure Boot off
# Import ggml-htp-v1.cer into both stores via certlm.msc:
#   - Trusted Root Certification Authorities
#   - Trusted Publishers
```

See `third-party/llama.cpp/docs/backend/snapdragon/windows.md` for background.
