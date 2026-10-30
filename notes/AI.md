# AI assistants

## Claude Code

| Path | Purpose |
|---|---|
| [`../CLAUDE.md`](../CLAUDE.md) | Project identity, hard constraints, and where each topic lives. Always in context — keep it short. |
| [`../.claude/settings.json`](../.claude/settings.json) | Read-only command allowlist. Reduces permission prompts without granting write access. |
| [`../.claude/skills/build`](../.claude/skills/build/SKILL.md) | `/build` — wraps [build.md](build.md). |
| [`../.claude/skills/release`](../.claude/skills/release/SKILL.md) | `/release` — wraps [release.md](release.md). Manual only (pushes tags). |
| [`../.claude/skills/geniex`](../.claude/skills/geniex/SKILL.md) | How to drive the `geniex` CLI non-interactively and check what actually ran. |

`third-party/geniex-qairt` ships its own `.claude/skills` for QAIRT runtime work; they load when editing files there.

### Rules for skills here

- A skill states intent and the judgment calls only. Commands and facts live in one canonical doc (`notes/` for developers, `docs/` for users), and the skill links to it.
- Keep each `SKILL.md` under ~50 lines. Nothing machine-, person-, or incident-specific.
- If a skill and its doc disagree, the doc wins. Fix the skill in the same PR.
- Personal workflow (issue trackers, accounts, device access) belongs in your own `~/.claude`, not here.
