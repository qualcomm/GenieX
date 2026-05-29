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

"""qairt LLM matrix: npu (only supported backend for qairt)."""

from __future__ import annotations

import geniex
import pytest

from _models import QAIRT_LLM_MODEL

pytestmark = pytest.mark.llm


@pytest.mark.parametrize('device_map', ['npu'])
def test_generate_blocking(qairt_llm_paths, device_map):
    with geniex.AutoModelForCausalLM.from_pretrained(
        QAIRT_LLM_MODEL,
        device_map=device_map,
    ) as llm:
        assert isinstance(llm, geniex.GenieXLLM)
        out = llm.generate('Say hi.', max_new_tokens=16, temperature=0.0, seed=42)
        assert isinstance(out, geniex.GenerateOutput)
        assert out.text
        assert out.profile.generated_tokens > 0
