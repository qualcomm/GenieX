#!/usr/bin/env python3
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Render the Breeze grading payload from a list of (prompt, response) items.

The payload is one self-contained prompt: the grade rules followed by the
numbered items to grade. It is uploaded as a workflow artifact rather than
inlined into the trigger issue because GitHub caps an issue body at 65536
characters and a real quality sweep blows past that.

Item source is a JSON file shaped `{"items": [{"prompt": ..., "response":
...}]}`. `sample-items.json` is a placeholder standing in for the real
on-device generation output until that leg is wired up.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent


def render(rules: str, items: list[dict[str, str]]) -> str:
    out = [rules.strip(), '']
    for i, item in enumerate(items):
        out += [
            f'========== ITEM {i} ==========',
            '',
            '--- Prompt ---',
            item['prompt'].strip(),
            '--- Response to grade ---',
            item['response'].strip(),
            '--- End of response ---',
            'Rationale:',
            '',
        ]
    out += [
        f'Grade all {len(items)} items above. Report a markdown table with the',
        'columns: Item | Prompt | Rationale | Rating. Put a bare integer in the',
        'Rating column.',
    ]
    return '\n'.join(out) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--items', type=pathlib.Path, default=HERE / 'sample-items.json')
    ap.add_argument('--rules', type=pathlib.Path, default=HERE / 'grade-rules.md')
    ap.add_argument('--out', type=pathlib.Path, default=pathlib.Path('grade-payload.md'))
    args = ap.parse_args()

    items = json.loads(args.items.read_text(encoding='utf-8'))['items']
    if not items:
        print(f'ERROR: no items in {args.items}', file=sys.stderr)
        return 1
    for i, item in enumerate(items):
        missing = {'prompt', 'response'} - item.keys()
        if missing:
            print(f'ERROR: item {i} missing {sorted(missing)}', file=sys.stderr)
            return 1

    payload = render(args.rules.read_text(encoding='utf-8'), items)
    args.out.write_text(payload, encoding='utf-8')
    print(f'{args.out}: {len(items)} items, {len(payload)} chars')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
