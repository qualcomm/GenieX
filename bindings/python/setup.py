# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Setuptools driver with install-time SDK download.

The sdist ships pure Python sources only. When pip (or `python -m build
--wheel`) assembles a wheel from the sdist, the custom `build_py` command
below runs `_sdk_fetch.fetch()` to download the platform-matching SDK zip
from GitHub Releases and stage its lib/ tree under geniex/lib/ so
package-data picks it up.

`python -m build --sdist` invokes the `sdist` command (not `build_py`), so
the tarball itself never contains prebuilt libs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

import _sdk_fetch  # noqa: E402


def _release_tag() -> str:
    """Read __release_tag__ from geniex/_version.py.

    Falls back to 'v' + __version__ when the tag attribute is absent so dev
    builds (where _version.py is just `__version__ = "0.1.0"`) still work.
    """
    ns: dict = {}
    exec((HERE / 'geniex' / '_version.py').read_text(), ns)
    return ns.get('__release_tag__', f'v{ns["__version__"]}')


RELEASE_TAG = _release_tag()


class BuildPyWithSdk(build_py):
    def run(self) -> None:
        pkg_dir = HERE / 'geniex'
        _sdk_fetch.fetch(pkg_dir, RELEASE_TAG)
        super().run()


setup(cmdclass={'build_py': BuildPyWithSdk})
