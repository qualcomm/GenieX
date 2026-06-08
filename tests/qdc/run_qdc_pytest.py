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

"""Run the SDK model-running pytest suite on a real QDC Android device.

GitHub runners only run the model-free ``api`` shard — they have no Snapdragon
hardware. The device-gated cells (``llama_cpp`` / ``qairt`` across every backend)
run here instead, on a QDC Android phone via the APPIUM framework:

  - this script packages pkg-geniex + the Python binding + the ``tests/`` tree
    into an artifact (under payload/) and submits it;
  - QDC runs the bundled appium pytest (``tests/qdc/appium/``) on its host, which
    builds a portable Python, deploys everything to the phone (pure-Python adb,
    see deploy.py — the host has no bash), runs the suite in ``adb shell``, and
    writes the JUnit XML back to the device's QDC_logs;
  - this script then downloads that XML and summarises it.

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
sys.path.insert(0, str(HERE / 'appium'))

import deploy  # noqa: E402 — stdlib-only; must follow the sys.path insert above

try:
    import _qdc
except ImportError:
    _qdc = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# tests/conftest.py resolves the VLM sample image relative to the repo root
# (parents[1] of the conftest), so the artifact must preserve this one asset.
TEST_IMAGE_REL = Path('cli/server/docs/ui/favicon-32x32.png')
# libggml-cpu.so needs libomp.so, which the CLI package omits but the Android
# app ships in extLibs; drop it beside the ggml libs (mirrors the benchmark).
OMP_REL = Path('bindings/android/app/extLibs/arm64-v8a/libomp.so')
_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.venv', '*.egg-info', 'models')


def build_android_artifact(pkg_dir: Path, tmp: Path) -> Path:
    # QDC's APPIUM framework runs `pytest tests` from the artifact root, so the
    # host-side appium harness (+ its pure-Python adb deploy) goes in tests/. The
    # mini-repo it deploys to the phone lives under payload/ so the host
    # collector never imports payload/tests/conftest.py (which imports geniex —
    # absent on the host); the harness points deploy.py at ../payload.
    stage = tmp / 'stage'
    payload = stage / 'payload'
    shutil.copytree(pkg_dir, payload / 'sdk' / 'pkg-geniex')
    shutil.copy(REPO / OMP_REL, payload / 'sdk' / 'pkg-geniex' / 'lib' / 'llama_cpp' / 'libomp.so')

    shutil.copytree(REPO / 'tests', payload / 'tests', ignore=_IGNORE)
    shutil.copytree(REPO / 'bindings' / 'python', payload / 'bindings' / 'python', ignore=_IGNORE)

    # Prebuild the Termux portable-Python usr/ tree here (the runner has public
    # internet; the QDC appium host is sandboxed and can't reach termux.dev).
    deploy.fetch_termux_usr(payload / 'termux-usr')

    img = payload / TEST_IMAGE_REL
    img.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / TEST_IMAGE_REL, img)

    host = stage / 'tests'
    host.mkdir()
    for name in ('conftest.py', 'test_run_suite.py', 'utils.py', 'deploy.py', 'requirements.txt'):
        shutil.copy(HERE / 'appium' / name, host / name)
    (stage / 'requirements.txt').write_text((HERE / 'appium' / 'requirements.txt').read_text())
    (stage / 'pytest.ini').write_text('[pytest]\naddopts = --junitxml=results.xml\n')

    return Path(shutil.make_archive(str(tmp / 'artifact'), 'zip', stage))


def summarise(xml: bytes) -> tuple[int, str]:
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
    lines = [
        '## QDC pytest — Android',
        '',
        f'**{verdict}** — {passed} passed, {failed} failed, {errored} errored, {skipped} skipped (of {total})',
        '',
    ]
    for status, name, msg, body in rows:
        if status == 'FAIL':
            lines += [f'<details><summary>{icon[status]} <code>{name}</code> — {msg}</summary>', '']
            lines += ['```', body, '```', '</details>']
        elif status == 'SKIP':
            lines.append(f'{icon[status]} `{name}` — {msg}')
        else:
            lines.append(f'{icon[status]} `{name}`')
    return (0 if verdict == 'PASS' else 1), '\n'.join(lines) + '\n'


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(path, 'a') as f:
            f.write(text)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pkg-dir', type=Path, required=True)
    p.add_argument('--device', default='SM8850')
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
        zip_path = build_android_artifact(args.pkg_dir, tmp)
        job_id = _qdc.submit_and_wait(
            client,
            target_id=target_id,
            job_name=f'geniex-pytest-{args.device}',
            platform='android',
            entry_script=None,
            zip_path=zip_path,
            timeout=args.job_timeout,
        )

        # Match on basename: QDC ships logs zipped and the inner member name
        # drops the directory prefix, so a path filter ("results/...") misses it.
        def _base(n):
            return n.replace('\\', '/').rsplit('/', 1)[-1]

        members = _qdc.download_log_members(client, job_id, tmp, lambda n: _base(n) == 'device-results.xml')
        # Always pull the device-side logs so the on-device run is visible in CI
        # regardless of pass/fail — harness.log (build/deploy/test), the appium
        # pytest stdout, and the pip install log. Skip the multi-MB logcat.
        diag = _qdc.download_log_members(
            client, job_id, tmp, lambda n: _base(n) in ('harness.log', 'appium_tests_stdout.txt', 'install.txt')
        )

    for name, data in diag:
        print(f'\n===== device log: {name} =====\n{data.decode("utf-8", "replace")}')

    if not members:
        log.error('no JUnit XML recovered (see device logs above)')
        write_summary('## QDC pytest — Android\n\nNo JUnit XML recovered (see device logs above).\n')
        return 1
    code, md = summarise(members[0][1])
    write_summary(md)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
