# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Shared driver for the three sibling sdists (#554).

Each sibling's setup.py is a 4-liner that calls ``run_setup(backends=...)``.
The Python sources, ``_sdk_fetch.py``, and this module are byte-identical
across all three sdists — ``bindings/sync_siblings.py`` mirrors them into
the sibling trees just before ``python -m build --sdist``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


def _release_tag(here: Path) -> str:
    ns: dict = {}
    exec((here / 'geniex' / '_version.py').read_text(), ns)
    return ns.get('__release_tag__', f'v{ns["__version__"]}')


def run_setup(*, here: Path, backends: Sequence[str]) -> None:
    """Drive ``setuptools.setup`` with an install-time SDK fetch.

    ``here`` is the directory of the calling ``setup.py`` (always
    ``Path(__file__).parent.resolve()``). ``backends`` selects which plugin
    subtrees the install-time fetcher stages — ``('llama-cpp', 'qairt')``
    for the meta sdist, a single-element tuple for each sibling.
    """
    sys.path.insert(0, str(here))
    import _sdk_fetch  # noqa: E402

    release_tag = _release_tag(here)

    class _BuildPyWithSdk(build_py):
        def run(self) -> None:
            _sdk_fetch.fetch(here / 'geniex', release_tag, backends=backends)
            super().run()

    setup(cmdclass={'build_py': _BuildPyWithSdk})
