# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Setuptools driver for the ``geniex-llama-cpp`` sibling sdist.

The shared logic lives in ``_shared_setup.run_setup``; this file picks the
backend tuple. See ``_shared_setup.py`` for details.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from _shared_setup import run_setup  # noqa: E402

run_setup(here=HERE, backends=('llama-cpp',))
