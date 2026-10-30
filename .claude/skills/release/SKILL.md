---
name: release
description: Propose and push the next SemVer release tag, then watch the release workflow.
argument-hint: "[tag]"
disable-model-invocation: true
---

# Release

[notes/release.md](../../../notes/release.md) defines the versioning rules and channels. Follow it. This skill only adds the guardrails.

1. **Gather facts.** Run `git fetch --tags --prune origin` first; stale local tags have caused wrong proposals. Find the latest stable tag with `git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1`. Read `git log <stable>..HEAD --no-merges --format='%h %s' --stat`. The working tree must be clean and in sync with origin.
2. **Propose one tag** per notes/release.md § Decision procedure, with a one-line reason. Ask only when two bumps are genuinely defensible. If the cycle holds only docs/chore commits, say so.
3. **Stable (`vX.Y.Z`) only:** the last rc's Windows SDK must be Microsoft-signed. Check the release body's "Hexagon HTP" section, or `gh release view <rc> --json assets` for a `-selfsigned` zip or `ggml-htp-v1.cer`. If it was self-signed, stop and point at notes/release.md § Hexagon HTP signing.
4. **Push only after the user approves this exact tag** in this session: `git tag <tag> && git push origin <tag>`.
5. **Watch**: `gh run watch <id> --exit-status`. A stable tag that already has release assets can't be re-run; cut a higher tag instead.

Never move, delete, or reuse a published tag. Pre-1.0 never bumps MAJOR.
