#!/usr/bin/env python3
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Mirror the canonical Python sources from ``bindings/python/`` into the
sibling sdist trees ``bindings/python-llama-cpp/`` and ``bindings/python-qairt/``,
and generate each sibling's ``pyproject.toml`` from the canonical one by
patching just the ``name`` and ``description`` fields.

Only ``setup.py`` is git-tracked under each sibling directory — the rest
(geniex/, _sdk_fetch.py, _shared_setup.py, _geniex_backend.py, README.md,
LICENSE, MANIFEST.in, pyproject.toml) is mirrored / generated here just
before ``python -m build --sdist``.

Run from the repo root before building any sibling sdist:

    python bindings/sync_siblings.py
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN = REPO_ROOT / 'bindings' / 'python'

# (sibling dir, sdist name, description) — only these two fields differ
# between the three pyproject.toml files; everything else is canonical.
SIBLINGS = (
    (
        'python-llama-cpp',
        'geniex-llama-cpp',
        'GenieX-Bridge Python binding (llama.cpp backend only) — install via pip install geniex-llama-cpp.',
    ),
    (
        'python-qairt',
        'geniex-qairt',
        'GenieX-Bridge Python binding (QAIRT backend only) — install via pip install geniex-qairt.',
    ),
)

COPY_FILES = (
    '_sdk_fetch.py',
    '_shared_setup.py',
    '_geniex_backend.py',
    'README.md',
    'LICENSE',
    'MANIFEST.in',
)
COPY_DIRS = ('geniex',)


def _patch_pyproject(canonical: str, *, name: str, description: str) -> str:
    out, count = re.subn(
        r'^name = "geniex"$',
        f'name = "{name}"',
        canonical,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit('canonical pyproject.toml: expected one `name = "geniex"` line')
    out, count = re.subn(
        r'^description = ".*"$',
        f'description = "{description}"',
        out,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit('canonical pyproject.toml: expected one `description = ...` line')
    return out


def sync_one(sibling_dir: str, name: str, description: str) -> None:
    sibling = REPO_ROOT / 'bindings' / sibling_dir
    if not sibling.exists():
        raise SystemExit(f'sibling tree missing: {sibling}')
    for entry in COPY_DIRS:
        dst = sibling / entry
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(MAIN / entry, dst)
    for entry in COPY_FILES:
        shutil.copy2(MAIN / entry, sibling / entry)

    canonical = (MAIN / 'pyproject.toml').read_text(encoding='utf-8')
    (sibling / 'pyproject.toml').write_text(
        _patch_pyproject(canonical, name=name, description=description),
        encoding='utf-8',
    )
    print(f'[sync] {sibling.relative_to(REPO_ROOT)} (name={name})')


def main() -> int:
    for sibling_dir, name, description in SIBLINGS:
        sync_one(sibling_dir, name, description)
    return 0


if __name__ == '__main__':
    sys.exit(main())
