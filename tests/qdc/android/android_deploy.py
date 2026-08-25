# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Deploy geniex + tests to a QDC Android phone and drive on-device pytest."""

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
    # QDC appium host's adb daemon drops connections; retry only on daemon errors.
    last = None
    for attempt in range(4):
        last = subprocess.run(['adb', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if last.returncode == 0:
            return last
        blob = last.stdout or ''
        if 'daemon' in blob or 'host.docker.internal' in blob:
            time.sleep(5 * (attempt + 1))
            continue
        break
    if check:
        assert last.returncode == 0, f'adb {args[0]} failed (exit {last.returncode}): {last.stdout}'
    return last


def adb_shell(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    # `adb shell` masks the remote exit code; the __RC__ sentinel recovers it.
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
    assert blob[:8] == b'!<arch>\n', 'not an ar archive'
    off = 8
    while off + 60 <= len(blob):
        header = blob[off : off + 60]
        name = header[:16].decode().strip().rstrip('/')
        size = int(header[48:58].decode().strip())
        start = off + 60
        yield name, blob[start : start + size]
        off = start + size + (size & 1)


def fetch_termux_usr(dest: Path) -> None:
    """Fetch Termux aarch64 debs and lay their usr/ tree at `dest`."""
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
                m.name = m.name.split('com.termux/files/usr/', 1)[1]
                if m.name:
                    members.append(m)
            t.extractall(dest, members=members)


def _push_tree(src: Path, dest: str, *, exclude: set[str] | None = None, include: set[str] | None = None) -> None:
    # One `adb push` per tree (tarballed) beats per-file pushes on flaky adb.
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


def _device_py_site() -> str:
    # Termux ships `python3` as a symlink to `python3.<N>` and puts wheels
    # under `lib/python3.<N>/site-packages/`. Rather than hard-coding the
    # minor version (Termux upgraded from 3.13 → 3.14 between drops), read
    # the actual dir off the device.
    out = adb_shell(f'ls {DEV_PY}/lib | grep -o "python3\\.[0-9]*" | head -1').stdout.strip()
    assert out.startswith('python3.'), f'no python3.X dir under {DEV_PY}/lib: {out!r}'
    return f'{DEV_PY}/lib/{out}/site-packages'


def deploy(payload: Path) -> None:
    if adb_shell(f'test -x {DEV_PY}/bin/python3', check=False).returncode != 0:
        # `adb push` from Windows preserves the file mode we set locally, and
        # tarfile.extractall on Windows can't preserve the source's exec bit —
        # so files land 0o666. Restore exec on the whole bin/lib tree in one
        # shot so python3 (a symlink) and its target actually run.
        adb('push', str(payload / 'termux-usr'), DEV_PY)
        adb_shell(f'chmod -R +x {DEV_PY}/bin {DEV_PY}/libexec 2>/dev/null; true')
        env = f'PREFIX={DEV_PY} LD_LIBRARY_PATH={DEV_PY}/lib HOME={DEV_ROOT} TMPDIR={DEV_ROOT}/tmp'
        adb_shell(f'mkdir -p {DEV_ROOT}/tmp')
        site = _device_py_site()
        adb_shell(f'{env} {DEV_PY}/bin/python3 -m pip install --target={site} pytest pytest-reportlog tqdm')

    _push_tree(payload / 'sdk' / 'pkg-geniex', DEV_SDK)
    adb_shell(
        f'cd {DEV_SDK}/lib && for f in qairt/htp-files/*.so qairt/htp-files/*.cat llama_cpp/*.so; do '
        f'bn=$(basename "$f"); [ "$bn" != libgeniex.so ] && [ -e "$f" ] && ln -sf "$f" "$bn"; done; true'
    )
    _push_tree(payload / 'tests', DEV_TESTS, exclude={'models', 'qdc'})
    _push_tree(payload / 'bindings' / 'python' / 'geniex', f'{_device_py_site()}/geniex')
    _push_tree(payload / 'cli', f'{DEV_ROOT}/cli', include={'server/docs/ui/favicon-32x32.png'})


def device_env() -> str:
    # libggml-base.so's DT_NEEDED for libomp.so resolves out of lib/llama_cpp/.
    lib_path = ':'.join(
        [
            f'{DEV_PY}/lib',
            f'{DEV_SDK}/lib',
            f'{DEV_SDK}/lib/llama_cpp',
            f'{DEV_SDK}/lib/qairt',
            f'{DEV_SDK}/lib/qairt/htp-files',
            '/system/lib64',
        ]
    )
    return (
        f'PREFIX={DEV_PY} '
        f'LD_LIBRARY_PATH={lib_path} '
        f'GENIEX_LIB_PATH={DEV_SDK}/lib '
        f'GENIEX_LOG=trace '
        f'HOME={DEV_ROOT} '
        f'GENIEX_DEVICE_TEST=1 '
        f'HF_HUB_DOWNLOAD_CONCURRENCY=1'
    )


def run_pytest(pytest_args: list[str]) -> int:
    # SDK routes plugin-scan errors through the Python `geniex` logger via a
    # C→Python callback; without a real handler+DEBUG level, the dlopen error
    # is dropped. Configure logging before geniex.init() to capture it.
    adb_shell(
        f'cd {DEV_TESTS} && {device_env()} {DEV_PY}/bin/python3 -c '
        '"import logging, geniex; '
        "logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s %(message)s'); "
        "geniex.init(); print('geniex.init OK')\"",
        check=False,
    )
    args = ' '.join(shlex.quote(a) for a in pytest_args)
    return adb_shell(
        f'cd {DEV_TESTS} && {device_env()} {DEV_PY}/bin/python3 -m pytest {args}',
        check=False,
    ).returncode
