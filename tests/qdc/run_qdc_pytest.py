# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the SDK pytest suite on a real QDC device (Linux QCS9075M / Windows SC8480XP)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'sdk' / 'benchmark' / 'qdc'))

try:
    import _qdc
except ImportError:
    _qdc = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# tests/conftest.py resolves this relative to the repo root, so preserve the path.
TEST_IMAGE_REL = Path('cli/server/docs/ui/favicon-32x32.png')
HTP_CERT_REL = Path('.github/certs/hexagon/ggml-htp-v1.cer')
_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '*.egg-info', 'models')


def _stage_common(pkg_dir: Path, stage: Path) -> None:
    shutil.copytree(pkg_dir, stage / 'pkg-geniex')
    shutil.copytree(REPO / 'tests', stage / 'tests', ignore=_IGNORE)
    shutil.copytree(REPO / 'bindings' / 'python', stage / 'bindings' / 'python', ignore=_IGNORE)
    img = stage / TEST_IMAGE_REL
    img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / TEST_IMAGE_REL, img)


def _build_windows_artifact(pkg_dir: Path, tmp: Path) -> Path:
    stage = tmp / 'stage'
    stage.mkdir()
    _stage_common(pkg_dir, stage)
    shutil.copy(REPO / HTP_CERT_REL, stage / 'ggml-htp-v1.cer')
    # CRLF — QDC's PowerShell parser is friendlier with CRLF.
    (stage / 'run_pytest.ps1').write_text((HERE / 'windows' / 'run_pytest.ps1').read_text(), newline='\r\n')
    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


def _build_linux_artifact(pkg_dir: Path, tmp: Path) -> Path:
    stage = tmp / 'stage'
    stage.mkdir()
    _stage_common(pkg_dir, stage)
    script_path = stage / 'run_pytest.sh'
    script_path.write_text((HERE / 'linux' / 'run_pytest.sh').read_text(), newline='\n')
    script_path.chmod(0o755)
    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


BUILDERS = {
    'linux': _build_linux_artifact,
    'windows': _build_windows_artifact,
}
ENTRY = {
    'linux': '/bin/bash /data/local/tmp/TestContent/run_pytest.sh',
    'windows': 'C:\\Temp\\TestContent\\run_pytest.ps1',
}


Row = tuple[str, str, str, str]


def _rows_from_junit(xml: bytes) -> list[Row]:
    root = ElementTree.fromstring(xml)
    suites = root.iter('testsuite') if root.tag != 'testsuite' else [root]
    rows: list[Row] = []
    for s in suites:
        for case in s.iter('testcase'):
            name = f'{case.get("classname", "")}::{case.get("name", "")}'
            fail = case.find('failure')
            err = case.find('error')
            skip = case.find('skipped')
            if fail is not None or err is not None:
                node = fail if fail is not None else err
                rows.append(('FAIL', name, node.get('message', ''), (node.text or '').strip()))
            elif skip is not None:
                rows.append(('SKIP', name, skip.get('message', ''), ''))
            else:
                rows.append(('PASS', name, '', ''))
    return rows


def _longrepr_text(longrepr: object) -> str:
    if not longrepr:
        return ''
    if isinstance(longrepr, str):
        return longrepr
    if isinstance(longrepr, dict):
        parts: list[str] = []
        tb = longrepr.get('reprtraceback') or {}
        for entry in tb.get('reprentries') or []:
            data = entry.get('data') or {}
            for ln in data.get('lines') or []:
                parts.append(ln)
            loc = data.get('reprfileloc') or {}
            if loc:
                path = loc.get('path', '')
                lineno = loc.get('lineno', '')
                msg = loc.get('message', '')
                parts.append(f'{path}:{lineno}: {msg}'.strip())
        crash = longrepr.get('reprcrash') or {}
        if crash and not parts:
            parts.append(f'{crash.get("path", "")}:{crash.get("lineno", "")}: {crash.get("message", "")}')
        return '\n'.join(parts).strip()
    return str(longrepr)


def _rows_from_reportlog(data: bytes) -> tuple[list[Row], set[str]]:
    outcomes: dict[str, Row] = {}
    started: set[str] = set()
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get('$report_type') != 'TestReport':
            continue
        nodeid = rec.get('nodeid') or ''
        when = rec.get('when')
        outcome = rec.get('outcome', '')
        if when == 'setup':
            started.add(nodeid)
        if when == 'call' or (when == 'setup' and outcome != 'passed'):
            body = _longrepr_text(rec.get('longrepr'))
            msg = ''
            crash = (rec.get('longrepr') or {}).get('reprcrash') if isinstance(rec.get('longrepr'), dict) else None
            if crash:
                msg = crash.get('message', '')
            if outcome == 'passed':
                outcomes[nodeid] = ('PASS', nodeid, '', '')
            elif outcome == 'skipped':
                outcomes[nodeid] = ('SKIP', nodeid, msg or body, '')
            else:
                outcomes[nodeid] = ('FAIL', nodeid, msg, body)
    incomplete = started - set(outcomes.keys())
    return list(outcomes.values()), incomplete


def _render_summary(rows: list[Row], label: str, incomplete: set[str] | None = None) -> tuple[int, str]:
    incomplete = incomplete or set()
    for nodeid in sorted(incomplete):
        rows.append(('FAIL', nodeid, 'process aborted before test completed', ''))
    passed = sum(1 for r in rows if r[0] == 'PASS')
    failed = sum(1 for r in rows if r[0] == 'FAIL')
    skipped = sum(1 for r in rows if r[0] == 'SKIP')
    total = passed + failed + skipped
    verdict = 'PASS' if failed == 0 else 'FAIL'
    icon = {'PASS': '✅', 'SKIP': '⏭️', 'FAIL': '❌'}
    lines = [
        f'## QDC Test — {label}',
        '',
        f'**{verdict}** — {passed} passed, {failed} failed, 0 errored, {skipped} skipped (of {total})',
        '',
    ]
    fails: list[tuple[str, str, str]] = []
    for status, name, msg, body in rows:
        if status == 'FAIL':
            lines.append(f'{icon[status]} `{name}`')
            fails.append((name, msg, body))
        elif status == 'SKIP':
            lines.append(f'{icon[status]} `{name}` — {msg}')
        else:
            lines.append(f'{icon[status]} `{name}`')
    if fails:
        # Model completions can contain $ / \n / unbalanced quotes; folding into
        # <details> stops GitHub from rendering $...$ as LaTeX and breaking the page.
        lines += ['', '### Failure details', '']
        for name, msg, body in fails:
            text = (body or msg or '').replace('\\n', '\n').replace('\\t', '\t')
            lines += [
                f'<details><summary><code>{name}</code></summary>',
                '',
                '```',
                text,
                '```',
                '</details>',
                '',
            ]
    return (0 if verdict == 'PASS' else 1), '\n'.join(lines) + '\n'


def summarise(xml: bytes, label: str = '') -> tuple[int, str]:
    return _render_summary(_rows_from_junit(xml), label)


def summarise_reportlog(data: bytes, label: str = '') -> tuple[int, str]:
    rows, incomplete = _rows_from_reportlog(data)
    return _render_summary(rows, label, incomplete)


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(path, 'a') as f:
            f.write(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pkg-dir', type=Path, required=True)
    p.add_argument('--platform', required=True, choices=sorted(BUILDERS))
    p.add_argument('--device', required=True, help='QDC device alias, e.g. QCS9075M / SC8480XP')
    p.add_argument('--job-timeout', type=int, default=10800)
    args = p.parse_args()

    if _qdc is None:
        raise SystemExit('qualcomm_device_cloud_sdk is required')
    api_key = os.environ.get('QDC_API_KEY')
    if not api_key:
        raise SystemExit('QDC_API_KEY must be set')

    label = f'{args.platform} ({args.device})'
    client = _qdc.make_client(api_key)
    target_id = _qdc.resolve_target(client, args.device)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = BUILDERS[args.platform](args.pkg_dir, tmp)
        job_id = _qdc.submit_and_wait(
            client,
            target_id=target_id,
            job_name=f'geniex-pytest-{args.platform}-{args.device}',
            platform=args.platform,
            entry_script=ENTRY[args.platform],
            zip_path=zip_path,
            timeout=args.job_timeout,
        )

        # JUnit is the clean-exit source of truth; --report-log NDJSON survives aborts.
        results = _qdc.download_log_members(
            client, job_id, tmp, lambda n: n in ('device-results.xml', 'device-report.log')
        )
        diag = _qdc.download_log_members(
            client,
            job_id,
            tmp,
            lambda n: n in ('harness.log', 'test_dbg.stdout', 'test.stdout', 'script.log'),
        )

    for name, data in diag:
        print(f'\n===== device log: {name} =====\n{data.decode("utf-8", "replace")}')

    by_name = {name: data for name, data in results}
    if b'</testsuite>' in by_name.get('device-results.xml', b''):
        code, md = summarise(by_name['device-results.xml'], label)
    elif by_name.get('device-report.log'):
        log.warning('JUnit XML missing or incomplete; reconstructing from --report-log NDJSON')
        code, md = summarise_reportlog(by_name['device-report.log'], label)
    else:
        log.error('no JUnit XML or report-log recovered (see device logs above)')
        write_summary(f'## QDC Test — {label}\n\nNo JUnit XML or report-log recovered.\n')
        return 1
    write_summary(md)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
