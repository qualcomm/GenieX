---
description: Cut and push a SemVer release tag following notes/release.md.
---

# /release

Cut a release tag. The authoritative rules — what each digit means,
what each channel means, and the decision procedure for picking the
next tag — live in [notes/release.md](../../notes/release.md). Read it
and follow it. This file only captures the operational loop and the
guardrails the agent must enforce.

## Flow

1. **Preconditions.** Working tree clean; branch up to date with its
   remote. Channel/branch constraints are listed in
   [notes/release.md](../../notes/release.md#what-each-channel-means) —
   check them. If any precondition fails, stop and tell the user.

2. **Gather the inputs the decision procedure needs.** Run these
   yourself — do not ask the user to paste output:
   - Latest tags (stable + pre-releases):
     `gh release list --limit 20`
   - Current branch: `git rev-parse --abbrev-ref HEAD`
   - Commits since the latest stable tag (or since repo root if none):
     `git log <latest-stable>..HEAD --oneline`
   - When a commit subject is ambiguous about whether it's a breaking
     change, read the diff for that commit
     (`git show --stat <sha>` then `git show <sha>` as needed).

3. **Apply the decision procedure from
   [notes/release.md](../../notes/release.md#decision-procedure-how-to-pick-the-next-tag)**
   to propose a tag. State the chosen tag plus a one-line reason
   (which bump, which channel, why). Only ask the user to choose if
   the commit range is genuinely ambiguous between two defensible
   bumps — don't punt routine decisions.

4. **Stable-tag extra check.** If the proposed tag is a bare
   `vX.Y.Z` (no channel suffix):
   - Confirm the prior `-rc.n` actually shipped green (check the
     release workflow run: `gh run list --workflow release.yml --limit 5`).
   - Confirm the SDK artifact from that run is **Microsoft-signed**,
     not self-signed (artifact name must not end in `-selfsigned`).
     If it is self-signed, **stop and warn the user**: stable tags
     require a signed HTP bundle. Point at the signing promotion
     steps in [notes/release.md](../../notes/release.md#promoting-self-signed--microsoft-signed).
   - If you cannot determine signing status, ask the user — do not
     proceed on assumption.

5. **Cut and push — requires explicit user approval.** Surface the
   exact commands and wait for confirmation:
   ```
   git tag <tag>
   git push origin <tag>
   ```

6. **Watch the release workflow** with `gh run watch`. Relay progress;
   if HTP flow branches (hit vs. miss), follow the handling in
   [notes/release.md](../../notes/release.md#hexagon-htp-signing).

## Guardrails

- **Never move or reuse a published tag.** If the wrong tag shipped,
  cut a higher one.
- **Never push a tag without the user's explicit go-ahead** for that
  specific tag in this session.
- **Never propose a bare stable tag off an unsigned build.** When in
  doubt about signing status, ask.
- **While the project is pre-1.0 (`X = 0`), never propose bumping
  `X`.** Breaking changes bump MINOR; see the Versioning section of
  the doc.

## If notes/release.md and this file disagree

Trust [notes/release.md](../../notes/release.md). Tell the user this
file is stale and suggest updating it.
