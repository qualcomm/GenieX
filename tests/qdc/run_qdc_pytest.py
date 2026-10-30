# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the SDK pytest suite on a real QDC device.

Two entry paths depending on the device family:
- Linux (QCS9075M) / Windows (SC8480XP): QDC executes a shell / PowerShell
  script from the artifact root that installs pytest, imports geniex, then
  runs the tests suite.
- Android (SM8850): QDC's APPIUM framework auto-discovers pytest under the
  artifact's tests/ on its own container host, which is what unlocks adb
  to the phone. The host-side test in `android/test_run_device.py` pushes
  a Termux-derived Python + pkg-geniex + tests/ to the phone and runs the
  device-gated pytest cells there.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree

HERE = Path(__file__).parent
REPO = HERE.parents[1]
MODELS_MANIFEST = REPO / 'tests' / 'models.json'
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
# PR smoke leaves CPU cells to GitHub runners and big models to nightly.
MARKEXPR = {
    'pr': '(llama_cpp or qairt) and not nightly and not device_cpu',
    'full': 'llama_cpp or qairt',
}
MARKEXPR_FILE = 'pytest-markexpr.txt'
_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '*.egg-info', 'models')


def _stage_common(pkg_dir: Path, stage: Path) -> None:
    shutil.copytree(pkg_dir, stage / 'pkg-geniex')
    shutil.copytree(REPO / 'tests', stage / 'tests', ignore=_IGNORE)
    shutil.copytree(REPO / 'bindings' / 'python', stage / 'bindings' / 'python', ignore=_IGNORE)
    img = stage / TEST_IMAGE_REL
    img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / TEST_IMAGE_REL, img)


def _build_windows_artifact(pkg_dir: Path, tmp: Path, markexpr: str) -> Path:
    stage = tmp / 'stage'
    stage.mkdir()
    _stage_common(pkg_dir, stage)
    (stage / MARKEXPR_FILE).write_text(markexpr)
    shutil.copy(REPO / HTP_CERT_REL, stage / 'ggml-htp-v1.cer')
    # CRLF — QDC's PowerShell parser is friendlier with CRLF.
    (stage / 'run_pytest.ps1').write_text((HERE / 'windows' / 'run_pytest.ps1').read_text(), newline='\r\n')
    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


def _build_linux_artifact(pkg_dir: Path, tmp: Path, markexpr: str) -> Path:
    stage = tmp / 'stage'
    stage.mkdir()
    _stage_common(pkg_dir, stage)
    (stage / MARKEXPR_FILE).write_text(markexpr)
    script_path = stage / 'run_pytest.sh'
    script_path.write_text((HERE / 'linux' / 'run_pytest.sh').read_text(), newline='\n')
    script_path.chmod(0o755)
    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


def _build_android_artifact(pkg_dir: Path, tmp: Path, markexpr: str) -> Path:
    # QDC APPIUM auto-discovers `pytest tests/` from the artifact root; the
    # payload the host-side driver pushes to the phone lives at payload/ so
    # the appium container's pytest never imports payload/tests/conftest.py
    # (which imports geniex — absent on the host).
    sys.path.insert(0, str(HERE / 'android'))
    import android_deploy  # noqa: PLC0415

    stage = tmp / 'stage'
    payload = stage / 'payload'
    shutil.copytree(pkg_dir, payload / 'sdk' / 'pkg-geniex')
    shutil.copytree(REPO / 'tests', payload / 'tests', ignore=_IGNORE)
    shutil.copytree(REPO / 'bindings' / 'python', payload / 'bindings' / 'python', ignore=_IGNORE)
    img = payload / TEST_IMAGE_REL
    img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / TEST_IMAGE_REL, img)
    # Fetched on the runner (public internet); appium container can't reach termux.dev.
    android_deploy.fetch_termux_usr(payload / 'termux-usr')

    host = stage / 'tests'
    host.mkdir()
    for name in ('conftest.py', 'test_run_device.py', 'android_deploy.py'):
        shutil.copy(HERE / 'android' / name, host / name)
    (stage / 'requirements.txt').write_text((HERE / 'android' / 'requirements.txt').read_text())
    (stage / MARKEXPR_FILE).write_text(markexpr)
    # -s (no capture) forces device-side pytest stdout through to the QDC stdout log —
    # otherwise host pytest swallows adb_shell output and only surfaces it on host FAIL.
    (stage / 'pytest.ini').write_text('[pytest]\naddopts = -s\n')
    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


BUILDERS = {
    'linux': _build_linux_artifact,
    'windows': _build_windows_artifact,
    'android': _build_android_artifact,
}
ENTRY = {
    'linux': '/bin/bash /data/local/tmp/TestContent/run_pytest.sh',
    'windows': 'C:\\Temp\\TestContent\\run_pytest.ps1',
    'android': None,
}


Row = tuple[str, str, str, str]

_STATUS_RE = re.compile(r'^(?:\[[^\]]+\]\s*)?(\S+\.py::.+?)\s+(PASSED|FAILED|SKIPPED|ERROR)\b(.*)$')
_TS_RE = re.compile(r'^\[[^\]]+\] ?')
_BANNER_RE = re.compile(r'^={3,} (.+?) ={3,}$')
_CASE_RE = re.compile(r'^_{3,} (?:ERROR at (?:setup|teardown) of )?(.+?) _{3,}$')


def _failure_bodies(text: str) -> dict[str, str]:
    # Map `name[params]` to its block in pytest's FAILURES / ERRORS section, minus DEBUG log lines.
    bodies: dict[str, list[str]] = {}
    in_failures = False
    current: str | None = None
    for raw in text.splitlines():
        line = _TS_RE.sub('', raw.rstrip(), count=1)
        if m := _BANNER_RE.match(line):
            in_failures = m.group(1) in ('FAILURES', 'ERRORS')
            current = None
        elif in_failures and (m := _CASE_RE.match(line)):
            current = m.group(1)
            bodies.setdefault(current, [])
        elif current and not line.startswith('DEBUG '):
            bodies[current].append(line)
    return {name: '\n'.join(lines).strip() for name, lines in bodies.items()}


def _rows_from_stdout(text: str) -> list[Row]:
    # Parse `test_file.py::name[params] STATUS [reason]` lines from pytest -v stdout.
    # A crash line (`Fatal Python error`, ggml `abort`) with no PASSED/FAILED trailing
    # the last-seen testcase is booked as that case's FAIL — pytest never got to write it.
    rows: list[Row] = []
    seen: dict[str, int] = {}
    last_incomplete: str | None = None
    crash_marker: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        m = _STATUS_RE.match(line)
        if m:
            name, status, tail = m.group(1), m.group(2), m.group(3).strip()
            if name in seen:
                continue
            seen[name] = len(rows)
            last_incomplete = None
            if status == 'PASSED':
                rows.append(('PASS', name, '', ''))
            elif status == 'SKIPPED':
                rows.append(('SKIP', name, tail.lstrip('(').rstrip(')'), ''))
            else:
                rows.append(('FAIL', name, tail, ''))
            continue
        # Testcase line without a trailing status → pytest started it but crashed mid-run.
        # Extract the nodeid: `file.py::name` optionally followed by `[params]` (params may contain spaces).
        m2 = re.match(r'^(?:\[[^\]]+\]\s*)?(\S+\.py::\S+?(?:\[[^\]]*\])?)(?:\s|$)', line)
        if m2 and m2.group(1) not in seen:
            last_incomplete = m2.group(1)
        if 'Fatal Python error' in line or 'ggml_abort' in line or 'ggml-hex:' in line:
            crash_marker = crash_marker or line
    if last_incomplete and crash_marker:
        rows.append(('FAIL', last_incomplete, crash_marker[:200], ''))
    bodies = _failure_bodies(text)
    return [
        (status, name, msg, body or bodies.get(name.split('::', 1)[-1].replace('::', '.'), ''))
        if status == 'FAIL'
        else (status, name, msg, body)
        for status, name, msg, body in rows
    ]


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


def _model_lines(tier: str) -> list[str]:
    try:
        with MODELS_MANIFEST.open(encoding='utf-8') as f:
            models = json.load(f).get('models', {})
    except (OSError, ValueError):
        return []
    if not models:
        return []
    lines = ['### Models', '', '| Slot | Model | Precision | Devices |', '|---|---|---|---|']
    for slot, entries in models.items():
        for entry in entries:
            if tier == 'pr' and entry.get('nightly'):
                continue
            env = entry.get('env_override')
            current_id = (env and os.environ.get(env)) or entry.get('id') or '—'
            prec = entry.get('precision') or '—'
            devices = ', '.join(entry.get('devices') or []) or '—'
            lines.append(f'| `{slot}` | `{current_id}` | `{prec}` | {devices} |')
    lines.append('')
    return lines


def _render_summary(
    rows: list[Row], label: str, incomplete: set[str] | None = None, tier: str = 'full'
) -> tuple[int, str]:
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
    lines += _model_lines(tier)
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


def summarise(xml: bytes, label: str = '', tier: str = 'full') -> tuple[int, str]:
    return _render_summary(_rows_from_junit(xml), label, tier=tier)


def summarise_reportlog(data: bytes, label: str = '', tier: str = 'full') -> tuple[int, str]:
    rows, incomplete = _rows_from_reportlog(data)
    return _render_summary(rows, label, incomplete, tier)


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
    p.add_argument('--tier', choices=sorted(MARKEXPR), default='full')
    p.add_argument('--job-timeout', type=int, default=10800)
    p.add_argument(
        '--logs-dir',
        type=Path,
        default=None,
        help='Persist device log files here for upload as a CI artifact.',
    )
    args = p.parse_args()

    if _qdc is None:
        raise SystemExit('qualcomm_device_cloud_sdk is required')
    api_key = os.environ.get('QDC_API_KEY')
    if not api_key:
        raise SystemExit('QDC_API_KEY must be set')

    label = f'{args.platform} ({args.device}) · {args.tier}'
    client = _qdc.make_client(api_key)
    target_id = _qdc.resolve_target(client, args.device)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = BUILDERS[args.platform](args.pkg_dir, tmp, MARKEXPR[args.tier])
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
            lambda n: n
            in (
                'harness.log',
                'test_dbg.stdout',
                'test.stdout',
                'script.log',
                'appium_tests_stdout.txt',
                'install.txt',
            ),
        )

    # Android APPIUM zips log members as `results/device-results.xml`; collapse
    # to basename so all three legs look up the same key.
    def _basename(n: str) -> str:
        return n.replace('\\', '/').rsplit('/', 1)[-1]

    if args.logs_dir:
        args.logs_dir.mkdir(parents=True, exist_ok=True)
        for name, data in list(results) + list(diag):
            (args.logs_dir / _basename(name)).write_bytes(data)

    for name, data in diag:
        # Encode via stdout.buffer so Windows cp1252 hosts don't blow up on
        # non-ASCII log bytes; CI's Linux runner is UTF-8 either way.
        header = f'\n===== device log: {name} =====\n'.encode()
        try:
            sys.stdout.buffer.write(header + data + b'\n')
        except AttributeError:
            print(header.decode() + data.decode('utf-8', 'replace'))

    by_name = {_basename(name): data for name, data in results}
    # Android's JUnit only covers the host-side wrapper test; per-case outcomes come from device stdout.
    if args.platform != 'android' and b'</testsuite>' in by_name.get('device-results.xml', b''):
        code, md = summarise(by_name['device-results.xml'], label, args.tier)
    elif by_name.get('device-report.log'):
        log.warning('JUnit XML missing or incomplete; reconstructing from --report-log NDJSON')
        code, md = summarise_reportlog(by_name['device-report.log'], label, args.tier)
    else:
        log.warning('no JUnit XML or report-log recovered; reconstructing from device stdout')
        diag_by_name = {_basename(name): data for name, data in diag}
        stdout_keys = ('appium_tests_stdout.txt', 'test.stdout', 'test_dbg.stdout')
        stdout_key = next((k for k in stdout_keys if k in diag_by_name), None)
        rows: list[Row] = []
        if stdout_key:
            rows = _rows_from_stdout(diag_by_name[stdout_key].decode('utf-8', 'replace'))
        if not rows:
            write_summary(f'## QDC Test — {label}\n\nNo test outcomes recovered.\n')
            return 1
        code, md = _render_summary(rows, label, tier=args.tier)
        write_summary(md)
        return code
    write_summary(md)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
