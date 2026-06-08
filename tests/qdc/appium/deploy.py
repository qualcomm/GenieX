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

"""Pure-Python adb deploy for the on-device pytest run (the #682 harness).

The QDC appium host is a minimal container with no bash, so the deploy logic
that lived in tests/android/*.sh is reimplemented here with adb + the Python
stdlib only (no bash, no ar, no curl). It builds a portable Python from Termux's
aarch64 .deb packages, pushes pkg-geniex + bindings + tests/ to the phone,
flattens the plugin libs, and runs the device-gated pytest cells in adb shell.
"""

from __future__ import annotations

import io
import shlex
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

DEV_ROOT = '/data/local/tmp'
DEV_PY = f'{DEV_ROOT}/termux-usr'
DEV_SDK = f'{DEV_ROOT}/geniex'
DEV_TESTS = f'{DEV_ROOT}/tests'

TERMUX_REPO = 'https://packages.termux.dev/apt/termux-main'
TERMUX_DEPS = (
    'python python-pip gdbm libandroid-posix-semaphore libandroid-support libbz2 '
    'libcrypt libexpat libffi liblzma libsqlite ncurses ncurses-ui-libs openssl '
    'readline zlib'
).split()


def adb(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # The QDC appium host reaches the device through an adb daemon at
    # host.docker.internal:5037 that occasionally drops a connection; retry a
    # few times before giving up so a transient timeout doesn't fail the run.
    last = None
    for attempt in range(4):
        last = subprocess.run(['adb', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if last.returncode == 0:
            return last
        if 'host.docker.internal' in (last.stdout or '') or 'daemon' in (last.stdout or ''):
            time.sleep(5 * (attempt + 1))
            continue
        break
    if check:
        assert last.returncode == 0, f'adb {args[0]} failed (exit {last.returncode}): {last.stdout}'
    return last


def adb_shell(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on-device via adb shell with an exit-code sentinel."""
    raw = subprocess.run(
        ['adb', 'shell', f'{cmd}; echo __RC__:$?'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rc, out = raw.returncode, raw.stdout or ''
    lines = out.rstrip('\n').split('\n')
    if lines and lines[-1].startswith('__RC__:'):
        rc = int(lines[-1][7:] or 0)
        out = '\n'.join(lines[:-1])
    print(out)
    if check:
        assert rc == 0, f'adb shell failed (exit {rc}): {cmd}'
    return subprocess.CompletedProcess(raw.args, rc, stdout=out)


def _ar_members(blob: bytes):
    """Yield (name, data) from a Unix ar archive (the .deb container format)."""
    assert blob[:8] == b'!<arch>\n', 'not an ar archive'
    off = 8
    while off + 60 <= len(blob):
        header = blob[off : off + 60]
        name = header[:16].decode().strip()
        size = int(header[48:58].decode().strip())
        start = off + 60
        yield name.rstrip('/'), blob[start : start + size]
        off = start + size + (size & 1)  # members are 2-byte aligned


def fetch_termux_usr(dest: Path) -> None:
    """Download Termux's aarch64 .deb packages and lay their usr/ tree at dest.

    dest becomes the portable-Python root (dest/bin/python3, dest/lib/...). Runs
    on the GitHub runner (which has public internet); the QDC appium host is a
    sandboxed container that can't reach packages.termux.dev, so the tree is
    fetched here and shipped in the artifact. Pure stdlib — no ar/curl needed.
    """
    index = urllib.request.urlopen(f'{TERMUX_REPO}/dists/stable/main/binary-aarch64/Packages').read().decode()
    filenames: dict[str, str] = {}
    pkg = None
    for line in index.splitlines():
        if line.startswith('Package: '):
            pkg = line[9:].strip()
        elif line.startswith('Filename: ') and pkg:
            filenames.setdefault(pkg, line[10:].strip())

    dest.mkdir(parents=True, exist_ok=True)
    for dep in TERMUX_DEPS:
        rel = filenames.get(dep)
        assert rel, f'package not in Termux index: {dep}'
        deb = urllib.request.urlopen(f'{TERMUX_REPO}/{rel}').read()
        data = next((d for n, d in _ar_members(deb) if n.startswith('data.tar')), None)
        assert data, f'no data.tar in {dep}'
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            members = []
            for m in t.getmembers():
                if 'com.termux/files/usr/' not in m.name:
                    continue
                m.name = m.name.split('com.termux/files/usr/', 1)[1]  # -> bin/..., lib/...
                if m.name:  # skip the bare usr/ dir entry
                    members.append(m)
            t.extractall(dest, members=members)


def deploy(repo_root: Path) -> None:
    # The Termux usr/ tree is prebuilt into the artifact (payload/termux-usr) by
    # the runner; push it and pip-install the test deps on-device (the phone has
    # internet to the pip mirror, the appium host does not).
    if adb_shell(f'test -x {DEV_PY}/bin/python3', check=False).returncode != 0:
        adb('push', str(repo_root / 'termux-usr'), DEV_PY)
        env = f'PREFIX={DEV_PY} LD_LIBRARY_PATH={DEV_PY}/lib HOME={DEV_ROOT} TMPDIR={DEV_ROOT}/tmp'
        site = f'{DEV_PY}/lib/python3.13/site-packages'
        adb_shell(f'mkdir -p {DEV_ROOT}/tmp')
        adb_shell(f'{env} {DEV_PY}/bin/python3 -m pip install --target={site} pytest tqdm')

    _push_tree(repo_root / 'sdk' / 'pkg-geniex', f'{DEV_SDK}')
    # Flatten plugin libs into lib/ (qairt loads GENIEX_LIB_PATH/libQnn*.so flat;
    # llama_cpp's Hexagon FastRPC resolves ggml-htp skels off a flat lib/).
    adb_shell(
        f'cd {DEV_SDK}/lib && for f in qairt/htp-files/*.so qairt/htp-files/*.cat llama_cpp/*.so; do '
        f'bn=$(basename "$f"); [ "$bn" != libgeniex.so ] && [ -e "$f" ] && ln -sf "$f" "$bn"; done; true'
    )
    _push_tree(repo_root / 'tests', DEV_TESTS, exclude={'models', 'qdc'})
    _push_tree(repo_root / 'bindings' / 'python' / 'geniex', f'{DEV_PY}/lib/python3.13/site-packages/geniex')
    # The VLM tests' conftest resolves the sample image at repo-root-relative
    # cli/server/docs/ui/; push it so the vlm cells run instead of skipping.
    _push_tree(repo_root / 'cli', f'{DEV_ROOT}/cli', include={'server/docs/ui/favicon-32x32.png'})


def _push_tree(src: Path, dest: str, exclude: set[str] | None = None, include: set[str] | None = None) -> None:
    """Push a directory tree to the device as a single tarball.

    One ``adb push`` per tree (vs one per file) keeps the transfer robust against
    the host's flaky adb daemon. ``exclude`` drops top-level entries; ``include``
    restricts to specific repo-relative paths (used to ship just one asset).
    """
    adb_shell(f'rm -rf {dest}')
    adb_shell(f'mkdir -p {dest}')
    with tempfile.TemporaryDirectory() as td:
        tarball = Path(td) / 'tree.tar.gz'
        with tarfile.open(tarball, 'w:gz') as t:
            if include is not None:
                for rel in sorted(include):
                    t.add(src / rel, arcname=rel)
            else:
                for child in sorted(src.iterdir()):
                    if exclude and child.name in exclude:
                        continue
                    t.add(child, arcname=child.name)
        adb('push', str(tarball), f'{dest}/tree.tar.gz')
    adb_shell(f'cd {dest} && tar -xzf tree.tar.gz && rm tree.tar.gz')


def device_env() -> str:
    return (
        f'PREFIX={DEV_PY} '
        f'LD_LIBRARY_PATH={DEV_PY}/lib:{DEV_SDK}/lib:{DEV_SDK}/lib/qairt:{DEV_SDK}/lib/qairt/htp-files:/system/lib64 '
        f'GENIEX_LIB_PATH={DEV_SDK}/lib HOME={DEV_ROOT} GENIEX_DEVICE_TEST=1'
    )


def run_pytest(pytest_args: list[str]) -> int:
    args = ' '.join(shlex.quote(a) for a in pytest_args)
    return adb_shell(
        f'cd {DEV_TESTS} && {device_env()} {DEV_PY}/bin/python3 -m pytest {args}',
        check=False,
    ).returncode
