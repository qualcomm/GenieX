---
description: Cut and push a SemVer release tag following notes/release.md.
---

# /release

Cut a release tag. Authoritative rules live in
[notes/release.md](../../notes/release.md); this file inlines the
fast-path decision so the agent can propose a tag without reading the
notes. If they disagree, trust the notes and flag this file as stale.

## Fast-path flow

**Step 1 — Fetch remote tags + gather inputs in ONE bash call.** A
plain `git tag` without fetching can be stale and has caused
wrong-tag proposals. Merge commits add ~50% noise to `git log` with
no signal — strip them with `--no-merges`.

```bash
git fetch --tags --prune origin --quiet && \
echo "=== working tree ===" && git status --porcelain && \
echo "=== branch ===" && git status -sb | head -1 && \
STABLE=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1) && \
echo "=== latest stable === $STABLE" && \
echo "=== recent tags ===" && git tag --sort=-v:refname | head -15 && \
echo "=== commits since $STABLE ===" && \
  git log "$STABLE..HEAD" --no-merges --format="%h %s" --stat
```

Portability / gotchas:
- Pick the stable tag with `grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$'`.
  `grep -v -- '-'` crashes on ugrep (treats `-` as stdin, not a
  pattern). The positive regex also rules out stray `v0.1` / `v1`
  tags.
- `git describe` only walks HEAD's ancestor chain and misses stable
  tags cut on side branches — don't use it.
- Empty `$STABLE` (new repo) → target `v0.1.0` and skip to step 3.

Preconditions that must hold before proceeding:
- working tree clean (untracked `.claude/worktrees/` is fine);
- branch in sync with `origin/<branch>` — `git status -sb` shows no
  `ahead`/`behind`;
- current branch matches channel rule: `alpha.n` → feature branch or
  main; `beta.n`, `rc.n`, stable → `main` only.

**Step 2 — Propose a tag.** Apply in order:

1. **Target `X.Y.Z`** from the `git log` output above:
   - breaking change (CLI flag removed/renamed, header signature
     change, Python API removed, config key renamed) → `v0.(A+1).0`
     — **pre-1.0: breaking bumps MINOR, never MAJOR**;
   - else any feature → `v0.(A+1).0`;
   - else fix/docs/chore/perf → `v0.A.(B+1)`.

   Conventional-Commits prefixes are hints, not contracts. The
   `--stat` output usually settles breaking-change questions; only
   `git show <sha>` when subject + stat are still ambiguous.

2. **Channel** by cycle position for that target:
   - first tag toward a new target, feature branch → `-alpha.1`;
   - first tag toward a new target, `main` → `-rc.1` (skip beta unless
     the user asks);
   - already in cycle → advance `n` (`rc.1` → `rc.2`) or move one
     channel forward (`beta.3` → `rc.1`, reset `n`);
   - all `-rc.n` green **and** last build Microsoft-signed → bare
     `vX.Y.Z`.

3. **Mid-cycle breaking change** that raises the target: abandon the
   current `X.Y.Z` (leave tags in place), restart at
   `v0.(new)-alpha.1` or `-rc.1` per branch.

State the proposed tag + a one-line reason (bump + channel + why).
Only ask the user to pick when two bumps are genuinely defensible.
**If the cycle since the latest stable is only one or two
docs/chore/style commits, flag that** — the user may want to wait
rather than spin an `-rc.1` on trivial noise.

**Step 3 — Stable-tag extra check (only if the proposed tag is bare
`vX.Y.Z`).** Skip entirely for pre-release tags.

```bash
gh run list --workflow release.yml --limit 5 --json name,conclusion,headSha,displayTitle
```

- Prior `-rc.n` must be green.
- SDK artifact name must **not** end in `-selfsigned`.
- Self-signed → stop, warn, point at
  [notes/release.md § Promoting self-signed → Microsoft-signed](../../notes/release.md#promoting-self-signed--microsoft-signed).
- Cannot determine → ask; don't assume.

**Step 4 — Cut and push (requires explicit user approval for the
specific tag in this session).**

```bash
git tag <tag> && git push origin <tag>
```

**Step 5 — Watch the run.** `gh run watch` (no sleep loops). On HTP
miss/hit, follow
[notes/release.md § Hexagon HTP signing](../../notes/release.md#hexagon-htp-signing).

## Guardrails

- **Never move or reuse a published tag.** Wrong tag shipped → cut a
  higher one.
- **Never push a tag without the user's explicit go-ahead** for that
  specific tag in this session.
- **Never propose a bare stable tag off an unsigned build.** Unsure →
  ask.
- **Pre-1.0 (`X = 0`): never bump `X`.** Breaking bumps MINOR.
