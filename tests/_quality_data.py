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
scorecard (`scripts/snapdragon/qdc/tests/run_scorecard_posix.py`). Prompts,
seed, n-predict, and substring-match logic stay aligned with upstream so a
regression on either side is comparable across the two suites.

Two intentional deltas vs. upstream:

- temperature is left at the SDK's `0.0 = defer-to-bundle-default` sentinel
  (see bindings/python/geniex/generation/config.py); the llama.cpp plugin
  resolves that to 0.8 (params.cpp). Upstream `llama-completion` similarly
  doesn't pass `--temp`, so it lands at the same plugin default — both sides
  sample, both rely on the matching `seed=1` for determinism. NOT greedy.
- VLM only ships the dog photo + first keyword set. Upstream also iterates a
  Qualcomm AIHub product image with vocabulary like person/phone/text; that
  second image is deferred to keep the in-repo asset surface small. Tracked
  alongside the perplexity follow-up.
"""

from __future__ import annotations

# (prompt, expected_substring). Matched case-insensitively against `out.text`.
LLM_QUALITY_PROMPTS: list[tuple[str, str]] = [
    ('The capital of France is', 'Paris'),
    ('2 + 2 =', '4'),
    ('The planet closest to the Sun is', 'Mercury'),
]

LLM_QUALITY_MAX_NEW_TOKENS = 256
LLM_QUALITY_TEMPERATURE = 0.0  # 0.0 = defer to plugin default; see module docstring
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
# Upstream uses 512; we cap at 256 — same headroom as the LLM cell, plenty of
# room for a keyword to appear in the caption, and bounded enough to keep the
# QDC Android wall-clock predictable across 4 VLM cells.
VLM_QUALITY_MAX_NEW_TOKENS = 256
VLM_QUALITY_TEMPERATURE = 0.0
VLM_QUALITY_SEED = 1
