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

"""SDK version metadata."""

from __future__ import annotations

import geniex


def test_version_nonempty(geniex_session):
    v = geniex.version()
    assert isinstance(v, str) and v


def test_qairt_runtime_version_nonempty(geniex_session):
    # Returns the QAIRT runtime version bundled with the SDK; available
    # even on hosts without an NPU because it is queried from the shipped
    # library, not the device.
    v = geniex.qairt_version()
    assert isinstance(v, str) and v


def test_llama_cpp_version_nonempty(geniex_session):
    v = geniex.llama_cpp_version()
    assert isinstance(v, str) and v
