---
description: Cut and push a SemVer release tag following docs/release.md.
---

# /release

Cut a release tag. The authoritative rules live in
[docs/release.md](../../docs/release.md); read it and follow it — do not
restate its rules here. This file only captures the operational loop and
the guardrails Claude must enforce.

## Flow

1. **Preconditions.** Confirm the working tree is clean and the branch
   is up to date with its remote. For stable / `-rc` / `-beta` tags the
   branch must be `master`; `-alpha.n` may be cut on a feature branch.
   If any precondition fails, stop and tell the user.
2. **Find the current latest tag** — `gh release list --limit 10`.
3. **Pick the next tag yourself.** Read
   [docs/release.md](../../docs/release.md) (especially the
   **Versioning** section) and inspect commits since the last tag with
   `git log <last-tag>..HEAD --oneline`. Apply the rules from the doc to
   decide MAJOR / MINOR / PATCH and the channel. State the chosen tag
   and a one-line reason before tagging. Only ask the user to choose if
   the commit range is genuinely ambiguous between two defensible bumps.
4. **Cut and push — requires explicit user approval.** Surface the exact
   `git tag <tag>` and `git push origin <tag>` commands and wait for
   confirmation. Tag pushes are irreversible.
5. **Watch the release workflow** with `gh run watch`. If the HTP signed
   bundle flow branches (hit vs. miss), follow the handling described
   in [docs/release.md](../../docs/release.md) and relay the operator
   steps back to the user.

## Guardrails

- **Never move or reuse a published tag.** If the wrong tag shipped,
  cut a higher one.
- **Never** push a tag without the user's explicit go-ahead for that
  specific tag in this session.

## If docs/release.md and this file disagree

Trust [docs/release.md](../../docs/release.md). Tell the user this file
is stale and suggest updating it.
