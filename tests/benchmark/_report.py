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

"""JSON / Markdown / stdout reporting for benchmark cells.

Schema is versioned (`schema_version: 1`) so a follow-up CI runner can ingest
the JSON without parsing the Markdown. Bump the version on any breaking shape
change and document the bump in tests/benchmark/README.md.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import geniex

SCHEMA_VERSION = 1


def _git_sha_short() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or 'unknown'
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _host_info(snapdragon: bool) -> dict[str, Any]:
    return {
        'os': platform.system(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'snapdragon': snapdragon,
    }


def build_report(
    cells: list[dict[str, Any]],
    *,
    snapdragon: bool,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        'schema_version': SCHEMA_VERSION,
        'generated_at': _dt.datetime.now(_dt.timezone.utc).isoformat(timespec='seconds'),
        'git_sha': _git_sha_short(),
        'geniex_version': getattr(geniex, '__version__', 'unknown'),
        'host': _host_info(snapdragon),
        'params': params,
        'cells': cells,
    }


def report_basename(report: dict[str, Any]) -> str:
    ts = report['generated_at'].replace(':', '').replace('-', '').replace('+0000', 'Z')
    return f'{ts}-{report["git_sha"]}'


def write_json(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{report_basename(report)}.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return path


def _fmt(value: float, fmt: str = '{:.1f}') -> str:
    if not value:
        return '-'
    return fmt.format(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f'# Benchmark — {report["git_sha"]}')
    lines.append('')
    lines.append(f'- Generated: `{report["generated_at"]}`')
    lines.append(f'- geniex version: `{report["geniex_version"]}`')
    h = report['host']
    lines.append(f'- Host: `{h["os"]} {h["machine"]}` (Snapdragon: `{h["snapdragon"]}`)')
    p = report['params']
    lines.append(
        f'- Params: warmup={p["warmup"]}, repeat={p["repeat"]}, '
        f'max_new_tokens={p["max_new_tokens"]}, temperature={p["temperature"]}, seed={p["seed"]}'
    )
    lines.append('')

    cells_by_model: dict[str, list[dict[str, Any]]] = {}
    for cell in report['cells']:
        cells_by_model.setdefault(cell['model'], []).append(cell)

    for model, group in cells_by_model.items():
        lines.append(f'## `{model}`')
        lines.append('')
        lines.append(
            '| Backend    | Device  | Status   | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Gen tokens | Notes |'
        )
        lines.append(
            '|------------|---------|----------|-----------|------------------|-----------------|------------|-------|'
        )
        for cell in group:
            status = cell['status']
            note = ''
            if status == 'skipped':
                ttft = prefill = decode = gen = '-'
                note = cell['skip_reason'] or ''
            elif status == 'error':
                ttft = prefill = decode = gen = '-'
                note = cell['error'] or 'error'
            else:
                agg = cell['agg']
                ttft = _fmt(agg['ttft_ms']['median'])
                prefill = _fmt(agg['prefill_tps']['median'])
                decode = _fmt(agg['decode_tps']['median'])
                gen = str(agg['gen_tokens']['median'])
            lines.append(
                f'| {cell["backend"]:<10} | {cell["device"]:<7} | {status:<8} | '
                f'{ttft:>9} | {prefill:>16} | {decode:>15} | {gen:>10} | {note} |'
            )
        lines.append('')
    return '\n'.join(lines)


def write_markdown(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{report_basename(report)}.md'
    path.write_text(render_markdown(report), encoding='utf-8')
    return path


def stdout_summary_line(cell: dict[str, Any]) -> str:
    if cell['status'] == 'skipped':
        return f'[skip] {cell["cell_id"]}  {cell["skip_reason"]}'
    if cell['status'] == 'error':
        return f'[err ] {cell["cell_id"]}  {cell["error"]}'
    agg = cell['agg']
    return (
        f'[ok  ] {cell["cell_id"]}  '
        f'ttft={_fmt(agg["ttft_ms"]["median"])} ms  '
        f'prefill={_fmt(agg["prefill_tps"]["median"])} tps  '
        f'decode={_fmt(agg["decode_tps"]["median"])} tps  '
        f'gen={agg["gen_tokens"]["median"]} tok'
    )
