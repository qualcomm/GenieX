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


def test_qairt_plugin_version_nonempty(geniex_session):
    # Plugin reports its own version; available on hosts without an NPU
    # because the value comes from the shipped library, not the device.
    v = geniex.get_plugin_version('qairt')
    assert isinstance(v, str) and v


def test_llama_cpp_plugin_version_nonempty(geniex_session):
    v = geniex.get_plugin_version('llama_cpp')
    assert isinstance(v, str) and v
