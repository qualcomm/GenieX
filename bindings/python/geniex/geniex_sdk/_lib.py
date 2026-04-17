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

import ctypes
import ctypes.util
import os
import sys

_lib: ctypes.CDLL | None = None


def _lib_name() -> str:
    if sys.platform == 'win32':
        return 'geniex.dll'
    elif sys.platform == 'darwin':
        return 'libgeniex.dylib'
    else:
        return 'libgeniex.so'


def load_library() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib

    name = _lib_name()
    candidates: list[str] = []

    # 1. Explicit override via env var
    env_path = os.environ.get('GENIEX_LIB_PATH')
    if env_path:
        candidates.append(env_path)

    # 2. Beside this package (Bazel runfiles place the .so here)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(pkg_dir, name))

    # 3. Parent directories (pip install editable layout)
    for parent in (os.path.dirname(pkg_dir), os.path.dirname(os.path.dirname(pkg_dir))):
        candidates.append(os.path.join(parent, name))

    # 4. OS search path
    found = ctypes.util.find_library('geniex')
    if found:
        candidates.append(found)

    for path in candidates:
        if os.path.isfile(path):
            try:
                _lib = ctypes.CDLL(path)
                return _lib
            except OSError:
                continue

    raise OSError(
        f'Cannot find {name}. Set GENIEX_LIB_PATH to the library path, '
        'or place the library alongside the geniex package.'
    )
