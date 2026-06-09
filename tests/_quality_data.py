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

"""Quality-check prompts and image keywords shared by both plugin matrices.

Mirrors the LLM and VLM keyword checks from upstream test-llama.cpp's QDC
scorecard (`scripts/snapdragon/qdc/tests/run_scorecard_posix.py`). Sampler
params and substring-match logic stay in lock-step so a regression on either
side is comparable across the two suites.
"""

from __future__ import annotations

# (prompt, expected_substring). Matched case-insensitively against `out.text`.
LLM_QUALITY_PROMPTS: list[tuple[str, str]] = [
    ('The capital of France is', 'Paris'),
    ('2 + 2 =', '4'),
    ('The planet closest to the Sun is', 'Mercury'),
]

LLM_QUALITY_MAX_NEW_TOKENS = 256
LLM_QUALITY_TEMPERATURE = 0.0
LLM_QUALITY_SEED = 1

VLM_QUALITY_PROMPT = 'Describe this image in detail.'
VLM_QUALITY_KEYWORDS: tuple[str, ...] = (
    'dog',
    'puppy',
    'animal',
    'golden',
    'retriever',
    'grass',
    'outdoor',
    'pet',
)
VLM_QUALITY_MAX_NEW_TOKENS = 64
VLM_QUALITY_TEMPERATURE = 0.0
VLM_QUALITY_SEED = 1
