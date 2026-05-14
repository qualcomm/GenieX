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

"""Native library loader. Search order: ``GENIEX_LIB_PATH`` → wheel layout → in-tree dev build → OS linker."""

import ctypes
import ctypes.util
import logging
import os
import sys

_lib: ctypes.CDLL | None = None
_logger = logging.getLogger('geniex')


def _lib_name() -> str:
    if sys.platform == 'win32':
        return 'geniex.dll'
    elif sys.platform == 'darwin':
        return 'libgeniex.dylib'
    return 'libgeniex.so'


def _preload_shared_libs(directories: list[str]) -> None:
    # Pre-load sibling shared libs so they're already mapped when libgeniex
    # dlopens its plugins. LD_LIBRARY_PATH mutations after process start have
    # no effect on glibc's dlopen; pre-loading is the portable equivalent for
    # making DT_NEEDED resolution succeed in the wheel layout.
    #
    # Use RTLD_LOCAL (not RTLD_GLOBAL): the SDK's own plugin loader opens
    # each plugin with RTLD_LOCAL, and we must not promote plugin siblings
    # into the global symbol scope. Specifically, libggml-hexagon.so (in the
    # llama_cpp plugin) re-exports the FastRPC ABI as default-visibility
    # forwarders (remote_handle64_*, fastrpc_*, rpcmem_*, dspqueue_*) that
    # are NULL until htpdrv_init() runs. With RTLD_GLOBAL, those symbols
    # shadow libcdsprpc.so for any later-loaded module — including the QAIRT
    # plugin's libQnnHtpV*.so — and the first FastRPC call from QnnHtp
    # segfaults at PC=0. The Go CLI and Windows path don't have this problem
    # because neither uses a global-scope preload.
    flags = getattr(ctypes, 'RTLD_LOCAL', 0)

    def _so_files(d: str) -> list[str]:
        try:
            entries = os.listdir(d)
        except OSError:
            return []
        if sys.platform == 'win32':
            return [f for f in entries if f.endswith('.dll')]
        elif sys.platform == 'darwin':
            return [f for f in entries if '.dylib' in f]
        # Sorted so the unversioned libfoo.so loads before libfoo.so.0.9.11.
        return sorted(f for f in entries if '.so' in f and not f.endswith('.a'))

    loaded: set[str] = set()
    for d in directories:
        if not os.path.isdir(d):
            continue
        for fname in _so_files(d):
            full = os.path.join(d, fname)
            real = os.path.realpath(full)
            if real in loaded:
                continue
            try:
                ctypes.CDLL(full, flags)
                loaded.add(real)
            except OSError:
                pass  # wrong arch / missing dep / etc — skip


def _setup_env(lib_path: str, plugin_path: str, extra_dirs: list[str] | None = None) -> None:
    lib_dir = os.path.dirname(lib_path)

    preload_dirs = [lib_dir]
    try:
        for entry in os.scandir(lib_dir):
            if entry.is_dir():
                preload_dirs.append(entry.path)
    except OSError:
        pass
    if extra_dirs:
        preload_dirs.extend(extra_dirs)

    # On Windows add_dll_directory() must happen BEFORE LoadLibrary — PATH
    # changes don't affect DLL search on Windows 8+.
    if sys.platform == 'win32':
        for d in preload_dirs:
            if os.path.isdir(d):
                os.add_dll_directory(d)

    _preload_shared_libs(preload_dirs)

    key = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
    existing = os.environ.get(key, '')
    extra = os.pathsep.join(preload_dirs)
    os.environ[key] = f'{extra}{os.pathsep}{existing}' if existing else extra

    if not os.environ.get('GENIEX_PLUGIN_PATH'):
        os.environ['GENIEX_PLUGIN_PATH'] = plugin_path


def _find_release(name: str) -> tuple[str, str] | None:
    # site-packages/geniex/lib/{name, <plugin-subdirs>/}
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(pkg_root, 'lib', name)
    if os.path.isfile(lib_path):
        plugin_path = os.path.join(pkg_root, 'lib')
        return lib_path, plugin_path
    return None


def _find_dev(name: str) -> tuple[str, str] | None:
    # <repo>/sdk/pkg-geniex/lib/{name, <plugin-subdirs>/}
    current = os.path.dirname(os.path.abspath(__file__))
    repo_root: str | None = None
    for _ in range(10):
        if os.path.isdir(os.path.join(current, 'sdk')):
            repo_root = current
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    if repo_root is None:
        return None

    lib_path = os.path.join(repo_root, 'sdk', 'pkg-geniex', 'lib', name)
    if not os.path.isfile(lib_path):
        return None

    plugin_path = os.path.join(repo_root, 'sdk', 'pkg-geniex', 'lib')
    return lib_path, plugin_path


def load_library() -> ctypes.CDLL:
    """Load libgeniex and configure plugin / preload environment."""
    global _lib
    if _lib is not None:
        return _lib

    name = _lib_name()

    env_path = os.environ.get('GENIEX_LIB_PATH')
    if env_path:
        if os.path.isdir(env_path):
            env_path = os.path.join(env_path, name)
        if os.path.isfile(env_path):
            _setup_env(env_path, os.path.dirname(env_path))
            _lib = ctypes.CDLL(env_path)
            return _lib

    # If both layouts are present, pick the newer mtime so a stale wheel
    # copy doesn't shadow a fresh dev build (and vice versa). Warn once.
    release = _find_release(name)
    dev = _find_dev(name)

    if release and dev:
        release_mtime = os.path.getmtime(release[0])
        dev_mtime = os.path.getmtime(dev[0])
        if dev_mtime > release_mtime:
            _logger.warning(
                'Both release (%s) and dev (%s) native libraries exist; using dev (newer).',
                release[0],
                dev[0],
            )
            result = dev
        else:
            _logger.warning(
                'Both release (%s) and dev (%s) native libraries exist; using release (newer).',
                release[0],
                dev[0],
            )
            result = release
    else:
        result = release or dev

    if result:
        lib_path, plugin_path = result
        _setup_env(lib_path, plugin_path)
        _lib = ctypes.CDLL(lib_path)
        return _lib

    found = ctypes.util.find_library('geniex')
    if found:
        try:
            _lib = ctypes.CDLL(found)
            return _lib
        except OSError:
            pass

    raise OSError(
        f'Cannot find the geniex native library ({name}).\n'
        '\n'
        'The native library must be present alongside this package.\n'
        f'Set GENIEX_LIB_PATH=/path/to/lib/dir/ to point to your local build,\n'
        'or see https://github.qualcomm.com/qcom-it-nexa-ai/geniex for build instructions.'
    )
