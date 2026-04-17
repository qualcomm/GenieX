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

"""Shared library loader with automatic dev / release path discovery.

Search order
------------
1. ``GENIEX_LIB_PATH`` env var — explicit override, works in any environment.
2. **Release layout** — ``lib/libgeniex.so`` beside the ``geniex`` package root
   (i.e. ``site-packages/geniex/lib/libgeniex.so`` after ``pip install``).
3. **Dev layout** — walk up from this file to the repo root, then glob
   ``sdk/build-*/src/libgeniex.so`` and pick the most recently modified build.
4. OS linker (``ctypes.util.find_library``).

Side effects on success
-----------------------
* ``GENIEX_PLUGIN_PATH`` is set to the directory containing plugin subdirs so the
  C library's ``scan_plugins()`` can find them without manual configuration.
* The ``.so`` directory (and its immediate subdirectories) are prepended to
  ``LD_LIBRARY_PATH`` / ``PATH`` so transitive shared-lib deps resolve.
"""

import ctypes
import ctypes.util
import glob
import os
import sys

_lib: ctypes.CDLL | None = None


def _lib_name() -> str:
    if sys.platform == 'win32':
        return 'geniex.dll'
    elif sys.platform == 'darwin':
        return 'libgeniex.dylib'
    return 'libgeniex.so'


def _preload_shared_libs(directories: list[str]) -> None:
    """Pre-load shared libraries in *directories* into the process with RTLD_GLOBAL.

    ``os.environ['LD_LIBRARY_PATH']`` changes have no effect after the process
    has started.  Pre-loading deps with ``RTLD_GLOBAL`` exposes their symbols to
    subsequently loaded libraries, which is the portable in-process equivalent.

    Handles both cmake-install layout (versioned ``libfoo.so.0.9.11`` only) and
    in-tree build layout (unversioned ``libfoo.so`` symlinks present).
    Loads the *shortest* name for each base library so dlopen resolves sonames
    correctly (e.g. loads ``libfoo.so`` before ``libfoo.so.0.9.11``).
    """
    flags = getattr(ctypes, 'RTLD_GLOBAL', 0x100)  # RTLD_GLOBAL = 0x100 on Linux

    def _so_files(d: str) -> list[str]:
        try:
            entries = os.listdir(d)
        except OSError:
            return []
        if sys.platform == 'win32':
            return [f for f in entries if f.endswith('.dll')]
        elif sys.platform == 'darwin':
            return [f for f in entries if '.dylib' in f]
        else:
            # Include both libfoo.so and libfoo.so.0.9.11 — sort so unversioned loads first.
            return sorted(f for f in entries if '.so' in f and not f.endswith('.a'))

    loaded: set[str] = set()
    for d in directories:
        if not os.path.isdir(d):
            continue
        for fname in _so_files(d):
            full = os.path.join(d, fname)
            # Resolve symlinks to avoid loading the same inode twice.
            real = os.path.realpath(full)
            if real in loaded:
                continue
            try:
                ctypes.CDLL(full, flags)
                loaded.add(real)
            except OSError:
                pass  # ignore libs that fail (wrong arch, missing dep, etc.)


def _setup_env(lib_path: str, plugin_path: str, extra_dirs: list[str] | None = None) -> None:
    """Pre-load transitive deps and configure GENIEX_PLUGIN_PATH."""
    lib_dir = os.path.dirname(lib_path)

    # Dirs to pre-load: the lib's own dir, its immediate subdirs, plus any extra.
    preload_dirs = [lib_dir]
    try:
        for entry in os.scandir(lib_dir):
            if entry.is_dir():
                preload_dirs.append(entry.path)
    except OSError:
        pass
    if extra_dirs:
        preload_dirs.extend(extra_dirs)

    _preload_shared_libs(preload_dirs)

    # Also update LD_LIBRARY_PATH for any child processes the user might spawn.
    key = 'PATH' if sys.platform == 'win32' else 'LD_LIBRARY_PATH'
    existing = os.environ.get(key, '')
    extra = os.pathsep.join(preload_dirs)
    os.environ[key] = f'{extra}{os.pathsep}{existing}' if existing else extra

    # Set GENIEX_PLUGIN_PATH only if not already overridden by the user.
    if not os.environ.get('GENIEX_PLUGIN_PATH'):
        os.environ['GENIEX_PLUGIN_PATH'] = plugin_path


# ---------------------------------------------------------------------------
# Candidate finders
# ---------------------------------------------------------------------------


def _find_release(name: str) -> tuple[str, str] | None:
    """Return (lib_path, plugin_path) for the release (wheel-installed) layout.

    Expected structure after ``pip install``::

        site-packages/geniex/
            lib/
                libgeniex.so          ← lib_path
                llama_cpp/            ← plugin subdir scanned by C library
                    libgeniex_plugin.so
    """
    # This file is at: site-packages/geniex/geniex_sdk/_lib.py
    # geniex package root is two levels up.
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(pkg_root, 'lib', name)
    if os.path.isfile(lib_path):
        plugin_path = os.path.join(pkg_root, 'lib')
        return lib_path, plugin_path
    return None


def _find_dev(name: str) -> tuple[str, str, str] | None:
    """Return (lib_path, plugin_path) for the dev (in-repo build) layout.

    Walks up from this file to find the repo root (identified by the presence
    of ``sdk/`` directory), then globs ``sdk/build-*/src/<name>`` and picks
    the build with the most recent mtime.

    Expected build output structure::

        <repo>/
            sdk/
                build-default/
                    src/libgeniex.so   ← lib_path
                    plugins/           ← plugin_path (GENIEX_PLUGIN_PATH)
                    bin/               ← transitive .so deps
    """
    # Walk up to the repo root (has a "sdk/" child directory).
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

    pattern = os.path.join(repo_root, 'sdk', 'build-*', 'src', name)
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    # Pick the most recently modified build.
    lib_path = max(candidates, key=os.path.getmtime)
    build_dir = os.path.dirname(os.path.dirname(lib_path))  # sdk/build-*/
    plugin_path = os.path.join(build_dir, 'plugins')
    if not os.path.isdir(plugin_path):
        plugin_path = os.path.join(build_dir, 'bin')  # fallback

    # Also expose build_dir so _setup_env can add extra search dirs.
    return lib_path, plugin_path, build_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    name = _lib_name()

    # --- Priority 1: explicit env override ---
    env_path = os.environ.get('GENIEX_LIB_PATH')
    if env_path and os.path.isfile(env_path):
        # Trust the user; don't touch GENIEX_PLUGIN_PATH or LD_LIBRARY_PATH.
        _lib = ctypes.CDLL(env_path)
        return _lib

    # --- Priority 2: release (wheel) layout ---
    result = _find_release(name)
    if result:
        lib_path, plugin_path = result
        _setup_env(lib_path, plugin_path)
        _lib = ctypes.CDLL(lib_path)
        return _lib

    # --- Priority 3: dev (in-repo build) layout ---
    result = _find_dev(name)
    if result:
        lib_path, plugin_path, build_dir = result
        # Pass build_dir/bin as extra so transitive deps (libggml*.so) are pre-loaded.
        _setup_env(lib_path, plugin_path, extra_dirs=[os.path.join(build_dir, 'bin')])
        _lib = ctypes.CDLL(lib_path)
        return _lib

    # --- Priority 4: OS linker ---
    found = ctypes.util.find_library('geniex')
    if found:
        try:
            _lib = ctypes.CDLL(found)
            return _lib
        except OSError:
            pass

    raise OSError(
        f'Cannot find {name}.\n'
        '  Release: ensure the wheel was built with sdk/pkg-geniex/lib/ bundled.\n'
        '  Dev: build the SDK first with "cmake --build sdk/build-default".\n'
        '  Override: set GENIEX_LIB_PATH to the full path of the library.'
    )
