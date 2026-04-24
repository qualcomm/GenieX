# CLAUDE.md

## Cutting a release tag

When the user asks you to cut a release / bump the version / tag a version,
read [docs/release.md](docs/release.md) (especially the **Versioning** section)
and choose the tag yourself. Do NOT ask the user to pick the tag.

Steps:

1. Run `gh release list --limit 10` to see the current latest tag.
2. Read [docs/release.md](docs/release.md) so the tag you pick follows the
   documented rules (MAJOR/MINOR/PATCH bump triggers, alpha/beta/rc channel
   semantics, which branch each channel is tagged on).
3. Inspect recent commits since the last tag (`git log <last-tag>..HEAD
   --oneline`) to decide the bump and channel:
   - Breaking public-surface change → MAJOR (or MINOR while on `0.y.z`, and
     call it out in the PR description).
   - New feature / backend → MINOR.
   - Bug fix / dep / doc / CI → PATCH.
   - On a feature branch before merge → `-alpha.n`.
   - On `master`, pre-stable internal verification → `-rc.n`.
   - On `master`, going live → bare `vX.Y.Z`.
4. State the chosen tag and the one-line reason before tagging. Confirm with
   the user only if the commit range is ambiguous (e.g. a mix of breaking and
   non-breaking changes that could plausibly be MAJOR or MINOR).
5. `git tag <tag> && git push origin <tag>`.

Do not move or reuse published tags.
