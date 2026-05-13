---
description: Cut and push a SemVer release tag following notes/release.md.
---

# /release

Cut a release tag. The authoritative rules live in
[notes/release.md](../../notes/release.md) — trust that file if it
disagrees with this one. This skill inlines the fast-path decision so
the agent can usually propose a tag without reading the notes.

## Fast-path flow

**Step 1 — Preconditions + input gathering (run in parallel, single
message with multiple Bash calls).** Don't ask the user to paste output.

```bash
# 1) working tree clean?
git status --porcelain
# 2) current branch + upstream sync state
git rev-parse --abbrev-ref HEAD
git status -sb
# 3) latest stable + all recent tags (local, millisecond — avoids gh).
#    The first line is the latest stable; the rest covers pre-releases
#    for channel/counter lookup. Uses --sort so it's not limited to
#    HEAD's ancestor chain (git describe would miss stables on side
#    branches).
git tag --sort=-v:refname --list "v[0-9]*.[0-9]*.[0-9]*" | grep -v -- '-' | head -1
git tag --sort=-v:refname --list "v*" | head -30
# 5) commits since latest stable — subject + stat in ONE pass. Stat
#    alone resolves most breaking-change questions (which files moved).
#    Only run `git show <sha>` on a specific commit when subject+stat
#    are genuinely ambiguous.
git log <latest-stable>..HEAD --format="%h %s" --stat
```

If any precondition fails (dirty tree, branch behind remote, wrong
branch for the channel you're about to cut) — stop and tell the user.
Channel/branch constraints: `alpha.n` feature branch or main; `beta.n`,
`rc.n`, stable only on `main`.

**Step 2 — Propose a tag using the inlined decision table.** Apply in
order:

1. **Target `X.Y.Z`** from the log since latest stable `v0.A.B`:
   - breaking change (flag removed/renamed, header signature change,
     Python API removed, config key renamed) → `v0.(A+1).0` —
     **pre-1.0: breaking bumps MINOR, never MAJOR**;
   - else any feature → `v0.(A+1).0`;
   - else fix/docs/chore → `v0.A.(B+1)`.

   Conventional-Commits prefixes (`feat:`/`fix:`/`feat!:`) are hints,
   not contracts. The `--stat` output from step 1 usually settles
   ambiguous cases; only `git show <sha>` if it doesn't.

2. **Channel** by cycle position for that target:
   - first tag toward a new target, feature branch → `-alpha.1`;
   - first tag toward a new target, `main` → `-rc.1` (skip beta unless
     user asks);
   - already in cycle → advance `n` (`rc.1` → `rc.2`) or move one
     channel forward (`beta.3` → `rc.1`, reset `n`);
   - all `-rc.n` green **and** last build Microsoft-signed → bare
     `vX.Y.Z`.

3. **Mid-cycle breaking change** that raises the target: abandon the
   current `X.Y.Z` (leave tags in place), restart at
   `v0.(new)-alpha.1` or `-rc.1` per branch.

State the chosen tag + a one-line reason (bump + channel + why). Only
ask the user to pick when two bumps are genuinely defensible.

**Step 3 — Stable-tag extra check (only if proposed tag is bare
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
git tag <tag>
git push origin <tag>
```

**Step 5 — Watch the run.** `gh run watch` (no sleep loops). If HTP
branches hit/miss, follow
[notes/release.md § Hexagon HTP signing](../../notes/release.md#hexagon-htp-signing).

## Guardrails

- **Never move or reuse a published tag.** Wrong tag shipped → cut a
  higher one.
- **Never push a tag without the user's explicit go-ahead** for that
  specific tag in this session.
- **Never propose a bare stable tag off an unsigned build.** Unsure →
  ask.
- **Pre-1.0 (`X = 0`): never bump `X`.** Breaking bumps MINOR.

## If notes/release.md disagrees

Trust [notes/release.md](../../notes/release.md). Tell the user this
file is stale and suggest updating it.