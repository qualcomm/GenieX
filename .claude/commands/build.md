---
description: Build the geniex CLI, SDK bridge, or release installer.
---

# /build

Help the user build geniex. The authoritative reference is
[notes/build.md](../../notes/build.md); read it and follow it — do not
restate its commands here. This file only captures intent and the
decisions Claude should make.

## Flow

1. Ask the user which target, unless it is clear from context:
   - **Dev run** — run the CLI against a model (fast iteration).
   - **SDK bridge** — needed before any `local` SDK build; rebuild when
     C/C++ sources under `sdk/` change.
   - **Release installer** — the `.exe` shipped by CI; usually only for
     local smoke-testing of release packaging.
2. Read [notes/build.md](../../notes/build.md) and follow the section for
   that target and the user's platform.
3. Before running anything on Windows, confirm the preflight from
   [notes/build.md](../../notes/build.md) is satisfied (Developer Mode,
   symlink right, long paths; Hexagon toolchain path ≤250 chars).
4. If the user wants a `local` SDK run but `sdk/pkg-geniex/lib/` does
   not contain the bridge artifact for their platform, build the SDK
   bridge first, then do the dev run.

## If notes/build.md and this file disagree

Trust [notes/build.md](../../notes/build.md). Tell the user this file is
stale and suggest updating it.
