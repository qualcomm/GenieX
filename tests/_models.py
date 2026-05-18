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

"""Test-model identifiers for the SDK pytest matrix.

One model per (plugin, modality) cell — kept here (not in conftest) so test
modules can import the names directly without needing a package layout.
QAIRT entries honour ``GENIEX_QAIRT_MODEL`` / ``GENIEX_QAIRT_VLM_MODEL`` so
operators can swap in alternate models without editing the suite.
"""

from __future__ import annotations

import os

LLAMA_CPP_LLM_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLAMA_CPP_LLM_QUANT = 'Q4_0'
LLAMA_CPP_VLM_MODEL = 'ggml-org/SmolVLM-500M-Instruct-GGUF'

QAIRT_LLM_MODEL = os.environ.get('GENIEX_QAIRT_MODEL', 'qualcomm/Qwen3-4B-Instruct-2507')
QAIRT_VLM_MODEL = os.environ.get('GENIEX_QAIRT_VLM_MODEL', 'qualcomm/Qwen2.5-VL-7B-Instruct')
