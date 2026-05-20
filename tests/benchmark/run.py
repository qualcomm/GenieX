# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI entry point for the geniex inference benchmark.

Usage::

    python tests/benchmark/run.py                          # full matrix
    python tests/benchmark/run.py --backend llama_cpp      # filter by backend
    python tests/benchmark/run.py --device cpu             # filter by device
    python tests/benchmark/run.py --models qwen3-0.6b      # substring filter on model id
    python tests/benchmark/run.py --repeat 5 --warmup 2    # tune sample count
    python tests/benchmark/run.py --output-dir custom/     # override results dir
    python tests/benchmark/run.py --no-markdown            # skip Markdown output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# When invoked as `python tests/benchmark/run.py`, this file is __main__ so
# relative imports (`from ._matrix import ...`) don't work. Add the parent
# of `benchmark/` (i.e. `tests/`) to sys.path so the sibling modules and
# `_models.py` are both importable as flat modules.
_TESTS_ROOT = Path(__file__).resolve().parents[1]
_BENCH_DIR = Path(__file__).resolve().parent
for _p in (_TESTS_ROOT, _BENCH_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geniex  # noqa: E402
from _matrix import LLM_CELLS, filter_cells, is_snapdragon_host, skip_reason  # noqa: E402
from _report import (  # noqa: E402
    build_report,
    stdout_summary_line,
    write_json,
    write_markdown,
)
from _runner import MAX_NEW_TOKENS, SEED, TEMPERATURE, run_cell, skipped_cell  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog='tests.benchmark.run', description=__doc__)
    p.add_argument('--backend', choices=['llama_cpp', 'qairt'], default=None)
    p.add_argument('--device', choices=['cpu', 'gpu', 'npu', 'hybrid'], default=None)
    p.add_argument('--models', dest='model_substr', default=None, help='substring match on model id')
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    p.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'results',
    )
    p.add_argument('--no-markdown', action='store_true')
    p.add_argument('--no-json', action='store_true')
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    snapdragon = is_snapdragon_host()
    cells = filter_cells(
        LLM_CELLS,
        backend=args.backend,
        device=args.device,
        model_substr=args.model_substr,
    )
    if not cells:
        print('No cells match the given filters.', file=sys.stderr)
        return 2

    print(f'Host: snapdragon={snapdragon}  cells={len(cells)}')
    geniex.init()
    results: list[dict] = []
    try:
        for cell in cells:
            reason = skip_reason(cell, snapdragon)
            if reason:
                result = skipped_cell(cell, reason)
            else:
                result = run_cell(
                    cell,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    max_new_tokens=args.max_new_tokens,
                )
            results.append(result)
            print(stdout_summary_line(result), flush=True)
    finally:
        geniex.deinit()

    report = build_report(
        results,
        snapdragon=snapdragon,
        params={
            'warmup': args.warmup,
            'repeat': args.repeat,
            'max_new_tokens': args.max_new_tokens,
            'temperature': TEMPERATURE,
            'seed': SEED,
        },
    )

    if not args.no_json:
        path = write_json(report, args.output_dir)
        print(f'Wrote {path}')
    if not args.no_markdown:
        path = write_markdown(report, args.output_dir)
        print(f'Wrote {path}')

    has_error = any(c['status'] == 'error' for c in results)
    return 1 if has_error else 0


if __name__ == '__main__':
    raise SystemExit(main())
