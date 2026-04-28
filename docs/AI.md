# AI assistants in geniex

Minimal configuration so AI coding assistants can help on this repo without
rediscovering the build/release flow every time.

## Claude Code

### Quick use

- `/build` — build the CLI, the SDK bridge, or the release installer.
  Wraps [build.md](build.md).
- `/release` — cut and push a SemVer tag; watch the release workflow.
  Wraps [release.md](release.md).
- Contributing rules (commits, branches, PR format) live in
  [../CONTRIBUTING.md](../CONTRIBUTING.md).

Both commands are playbooks: they describe the steps Claude should take.
Trigger them with a leading slash in a Claude Code session.

### Layout

- [`CLAUDE.md`](../CLAUDE.md) — project identity + hard constraints. Always
  in Claude's context; keep it short.
- [`.claude/settings.json`](../.claude/settings.json) — read-only command
  allowlist (`git status`, `gh run list`, ...). Reduces permission prompts
  without granting write access.
- [`.claude/commands/*.md`](../.claude/commands/) — slash commands. Loaded
  on demand when the user types `/<name>`.
- [`.claude/skills/<name>/SKILL.md`](../.claude/skills/) — skills. Auto-load
  when the `description:` matches the user's request.

### How to extend

- **Add a command** — drop `.claude/commands/<name>.md` with a
  `description:` frontmatter line. Invoke with `/<name>`.
- **Add a skill** — create `.claude/skills/<name>/SKILL.md`. Skills
  auto-load based on a trigger description (e.g. "when editing files
  that import `foo`"). Prefer a command unless the knowledge should apply
  *without* being explicitly invoked.
- **Add a subagent** — create `.claude/agents/<name>.md`. Only worthwhile
  when a task recurs often enough to justify an isolated sub-context.

Rule of thumb: if you'd invoke it with a slash, it's a command. If it
should auto-apply while editing certain code, it's a skill. If it needs
its own context window, it's a subagent.

### What is intentionally *not* here

- No hooks — no confirmed "every time X → do Y" workflow yet.
- No subagents — no task yet recurs often enough to justify an isolated
  sub-context.

## GitHub Copilot

TBD.
