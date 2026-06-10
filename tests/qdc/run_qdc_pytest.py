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

"""Run the SDK model-running pytest suite on a real QDC device.

GitHub runners only run the model-free ``api`` shard — they have no Snapdragon
hardware. The device-gated cells (``llama_cpp`` / ``qairt`` across every backend)
run here instead, on a QDC X Elite Windows ARM64 device via the POWERSHELL
framework (``windows/run_pytest.ps1``):

  - this script packages pkg-geniex + the Python binding + the ``tests/`` tree
    + the HTP cert into an artifact and submits it;
  - QDC extracts the artifact under ``C:\\Temp\\TestContent\\`` and runs
    ``run_pytest.ps1`` there, which bootstraps a portable Python, installs
    pytest, then runs the suite directly against the windows-arm64 SDK;
  - the entry script writes the JUnit XML to ``C:\\Temp\\QDC_Logs\\`` and the
    POWERSHELL framework auto-uploads everything in QDC_Logs for collection.

The QDC submit/poll/collect plumbing is shared with the benchmark scorecard via
``sdk/benchmark/qdc/_qdc.py``.
"""

from __future__ import annotations

import argparse
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

# tests/conftest.py resolves the VLM sample image relative to the repo root
# (parents[1] of the conftest), so the artifact must preserve this asset.
TEST_IMAGE_REL = Path('cli/server/docs/ui/favicon-32x32.png')
HTP_CERT_REL = Path('.github/certs/hexagon/ggml-htp-v1.cer')
_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '*.egg-info', 'models')

ENTRY_SCRIPT = 'C:\\Temp\\TestContent\\run_pytest.ps1'


def build_windows_artifact(pkg_dir: Path, marker: str, tmp: Path) -> Path:
    """Stage everything QDC's POWERSHELL framework needs to run pytest on X Elite.

    The framework extracts the zip to ``C:\\Temp\\TestContent\\`` and invokes
    the entry script there; ``run_pytest.ps1`` boots a portable Python and runs
    pytest against the staged tree directly (no wheel install — sets
    ``PYTHONPATH=bindings\\python`` and ``GENIEX_LIB_PATH=pkg-geniex\\lib``,
    matching the windows-arm64 cell in ``.github/workflows/_test.yml``).

    ``marker`` is substituted into the pytest ``-m`` selector so the workflow
    can split the matrix across two jobs (one per plugin) and stay under
    QDC's 60-min POWERSHELL device timeout.
    """
    stage = tmp / 'stage'
    stage.mkdir()
    shutil.copytree(pkg_dir, stage / 'pkg-geniex')
    shutil.copytree(REPO / 'tests', stage / 'tests', ignore=_IGNORE)
    shutil.copytree(REPO / 'bindings' / 'python', stage / 'bindings' / 'python', ignore=_IGNORE)

    img = stage / TEST_IMAGE_REL
    img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / TEST_IMAGE_REL, img)

    shutil.copy(REPO / HTP_CERT_REL, stage / 'ggml-htp-v1.cer')
    # CRLF line endings — QDC's PowerShell parser is friendlier with CRLF.
    ps = (HERE / 'windows' / 'run_pytest.ps1').read_text().replace('{MARKER}', marker)
    (stage / 'run_pytest.ps1').write_text(ps, newline='\r\n')

    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


def summarise(xml: bytes, plugin: str = '') -> tuple[int, str]:
    """Parse JUnit XML; return (exit_code, markdown). Non-zero on any failure.

    Lists every cell with its status (like pytest's own per-test output) so it's
    clear which ran vs skipped, and folds each failure's traceback / each skip's
    reason in so the device-side detail is visible without re-running on QDC.
    """
    root = ElementTree.fromstring(xml)
    suites = root.iter('testsuite') if root.tag != 'testsuite' else [root]
    total = failed = errored = skipped = 0
    rows: list[tuple[str, str, str, str]] = []  # (status, name, message, body)
    for s in suites:
        total += int(s.get('tests', 0))
        failed += int(s.get('failures', 0))
        errored += int(s.get('errors', 0))
        skipped += int(s.get('skipped', 0))
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

    passed = total - failed - errored - skipped
    verdict = 'PASS' if failed == 0 and errored == 0 else 'FAIL'
    icon = {'PASS': '✅', 'SKIP': '⏭️', 'FAIL': '❌'}
    title_suffix = f' — {plugin}' if plugin else ''
    lines = [
        f'## QDC pytest — X Elite Windows{title_suffix}',
        '',
        f'**{verdict}** — {passed} passed, {failed} failed, {errored} errored, {skipped} skipped (of {total})',
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
        # Failure messages can include the full model completion (literal `\n`,
        # `$`, unbalanced quotes) — render inline and GitHub markdown will
        # parse `$...$` as LaTeX and break the page. Push everything into a
        # fenced code block under the per-test <details>; the test name stays
        # in the <summary> so the failure list above is one tidy line per id.
        # pytest's JUnit body already starts with the message, so prefer body
        # and fall back to message — printing both repeats the model output.
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


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(path, 'a') as f:
            f.write(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pkg-dir', type=Path, required=True)
    p.add_argument('--device', default='SC8380XP', help='QDC device alias (default: X Elite SC8380XP)')
    p.add_argument(
        '--plugin',
        required=True,
        help='Label for the matrix leg (used in job_name + summary heading).',
    )
    p.add_argument(
        '--pytest-marker',
        required=True,
        help='Pytest -m expression (e.g. "llama_cpp and llm") substituted into the entry script.',
    )
    p.add_argument('--job-timeout', type=int, default=10800)
    args = p.parse_args()

    if _qdc is None:
        raise SystemExit('qualcomm_device_cloud_sdk is required')
    api_key = os.environ.get('QDC_API_KEY')
    if not api_key:
        raise SystemExit('QDC_API_KEY must be set')

    client = _qdc.make_client(api_key)
    target_id = _qdc.resolve_target(client, args.device)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = build_windows_artifact(args.pkg_dir, args.pytest_marker, tmp)
        job_id = _qdc.submit_and_wait(
            client,
            target_id=target_id,
            job_name=f'geniex-pytest-{args.plugin}-{args.device}',
            platform='windows',
            entry_script=ENTRY_SCRIPT,
            zip_path=zip_path,
            timeout=args.job_timeout,
        )

        members = _qdc.download_log_members(client, job_id, tmp, lambda n: n == 'device-results.xml')
        # Always pull the device-side logs so the on-device run is visible in CI
        # regardless of pass/fail. test_dbg.stdout is QDC's full PowerShell
        # stdout / stderr capture — strictly more complete than our
        # Start-Transcript harness.log, which buffers and can drop tail bytes
        # when the framework aborts the process on its 60-min device timeout.
        diag = _qdc.download_log_members(
            client,
            job_id,
            tmp,
            lambda n: n in ('harness.log', 'test_dbg.stdout', 'test.stdout'),
        )

    for name, data in diag:
        print(f'\n===== device log: {name} =====\n{data.decode("utf-8", "replace")}')

    if not members:
        log.error('no JUnit XML recovered (see device logs above)')
        write_summary(
            f'## QDC pytest — X Elite Windows — {args.plugin}\n\n' 'No JUnit XML recovered (see device logs above).\n'
        )
        return 1
    code, md = summarise(members[0][1], args.plugin)
    write_summary(md)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
