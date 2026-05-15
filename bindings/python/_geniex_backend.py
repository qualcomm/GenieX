# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PEP 517 wrapper around setuptools.build_meta.

Qualcomm Linux ships a stripped-down Python 3.12 without ``tomllib`` in
stdlib. ``setuptools.compat.py310`` imports ``tomllib`` unconditionally
on Python 3.11+ — there is no upstream tomli fallback on that branch —
so the isolated build env explodes before our ``setup.py`` runs.

This wrapper registers ``tomli`` under the stdlib name first, then
delegates every PEP 517 / PEP 660 hook to ``setuptools.build_meta``
verbatim. On hosts that already have stdlib ``tomllib`` the alias path
is skipped and behaviour matches plain setuptools. See GH #538.
"""

import sys

if 'tomllib' not in sys.modules:
    try:
        import tomllib  # noqa: F401  — normal stdlib path on Python 3.11+
    except ModuleNotFoundError:
        import tomli

        sys.modules['tomllib'] = tomli

from setuptools.build_meta import (  # noqa: E402,F401
    build_sdist,
    build_wheel,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
    prepare_metadata_for_build_wheel,
)

# PEP 660 editable hooks are only present in modern setuptools; re-export
# when available so `pip install -e .` keeps working unchanged.
try:
    from setuptools.build_meta import (  # noqa: E402,F401
        build_editable,
        get_requires_for_build_editable,
        prepare_metadata_for_build_editable,
    )
except ImportError:
    pass
