# AI assistants

Minimal configuration so AI coding assistants can operate on this repo without rediscovering the build/release flow each session.

## Claude Code

### Slash commands

- `/build` — build the CLI, the SDK bridge, or the release installer. Playbook wraps [build.md](build.md).
- `/release` — cut and push a SemVer tag, then watch the release workflow. Playbook wraps [release.md](release.md).

Contributing rules (commits, branches, PR format) live in [../CONTRIBUTING.md](../CONTRIBUTING.md).

### Layout

| Path                                                       | Purpose                                                                  |
|------------------------------------------------------------|--------------------------------------------------------------------------|
| [`../CLAUDE.md`](../CLAUDE.md)                             | Project identity + hard constraints. Always in context — keep it short. |
| [`../.claude/settings.json`](../.claude/settings.json)     | Read-only command allowlist (`git status`, `gh run list`, …). Reduces permission prompts without granting write access. |
| [`../.claude/commands/*.md`](../.claude/commands/)         | Slash commands. Loaded on demand when the user types `/<name>`.         |
| [`../.claude/skills/<name>/SKILL.md`](../.claude/skills/)  | Skills. Auto-load when the `description:` matches the user's request.   |

### Extending

- **Command** — drop `.claude/commands/<name>.md` with a `description:` frontmatter line. Invoke with `/<name>`.
- **Skill** — create `.claude/skills/<name>/SKILL.md`. Auto-loads based on the trigger description (e.g. "when editing files that import `foo`"). Prefer a command unless the knowledge should apply *without* being explicitly invoked.
- **Subagent** — create `.claude/agents/<name>.md`. Only worthwhile when a task recurs often enough to justify an isolated sub-context.

Rule of thumb: slash invocation → command; auto-apply while editing certain code → skill; own context window → subagent.

### Intentionally absent

- **Hooks** — no confirmed "every time X → do Y" workflow yet.
- **Subagents** — no task recurs often enough yet to justify an isolated sub-context.

## GitHub Copilot

Not configured.
