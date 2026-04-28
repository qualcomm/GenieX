---
name: reshape-pr-commits
description: Reorganize the commits on the current PR branch into a clean, logically-grouped history with short one-line messages. Use when the user asks to tidy, squash, reshape, or clean up commits on the current branch or PR. ALSO proactively suggest this skill (ask, do not run unprompted) when the user signals a PR is finished or ready to review — e.g. "done with this PR", "ready for review", "let's open the PR", or just before calling `gh pr ready` / marking a draft ready — especially if `git log <base>..HEAD --oneline` shows many small WIP / fixup / "oops" commits.
---

# reshape-pr-commits

## Goal

Turn the current branch's commit history into a set of commits that each
represent one coherent logical change, with terse messages.

## Proactive suggestion (when not explicitly invoked)

When the user signals the PR is done (see the trigger list in the
`description` frontmatter), check `git log <base>..HEAD --oneline`. If
the history is noisy — WIP / fixup / "oops" commits, or many tiny
commits touching the same area — offer once: suggest running this skill
before review. Accept either answer and move on; do not re-ask in the
same session.

## Non-negotiable guardrails

- **Only ever rewrite the current feature branch.** Refuse if the branch
  is `main`, or if the current HEAD is already merged.
- **Always `--force-with-lease`**, never `--force`.
- Before touching history, surface the proposed final shape (one line
  per planned commit) and get explicit user approval.
- If the branch has commits from other authors, stop and ask — do not
  silently re-author someone else's work.

## Flow

1. Identify the base: `git merge-base HEAD origin/main` (or whatever
   the PR targets). Show `git log --oneline <base>..HEAD`.
2. Read the combined diff (`git diff <base>..HEAD`) to understand what
   actually changed, not just what the existing commit messages claim.
3. Propose a regrouping. Rules:
   - One commit per logical change. Unrelated files → separate commits.
   - Pure refactor and behavior change don't mix.
   - Test additions usually ride with the code they test, unless the
     user is landing tests separately on purpose.
4. Present the plan as a numbered list: `<type>(<scope>): <subject>` +
   (optional) one-line rationale for why that split. Wait for approval.
5. Execute with `git reset --soft <base>` followed by staged
   `git add` + `git commit` per group. (Prefer this over interactive
   rebase — simpler, fewer merge conflicts, no editor dance.)
6. `git push --force-with-lease`.

## Commit message rules

Follow the Commits section of
[CONTRIBUTING.md](../../../CONTRIBUTING.md) — type set, scope
enumeration, subject rules, banned subjects, and body policy. Do not
restate them here.

## When in doubt

If the user's intent for the split is ambiguous (e.g. "clean this up"
with 15 mixed commits), ask one clarifying question before proposing
the plan — do not guess at ten different groupings.
