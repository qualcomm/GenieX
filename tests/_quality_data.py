# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Quality-check prompts and image keywords shared by both plugin matrices.

Mirrors the LLM and VLM keyword checks from upstream test-llama.cpp's QDC
scorecard (`scripts/snapdragon/qdc/tests/run_scorecard_posix.py`). Prompts,
seed, n-predict, sampler defaults, chat-template wrapping, and substring-match
logic stay aligned with upstream so a regression on either side is comparable
across the two suites.

Chat-template note: upstream invokes `llama-completion` without `-no-cnv`, so
`COMMON_CONVERSATION_MODE_AUTO` wraps the prompt in the model's chat template
(visible in the scorecard log as `chat template is available, enabling
conversation mode`). The test cases call `apply_chat_template` themselves
before `generate()` to reproduce that path — feeding the raw string lets
Qwen3-style models drift into completion mode and the keyword only appears
by sampler luck.

One intentional delta vs. upstream: VLM only ships the dog photo + first
keyword set. Upstream also iterates a Qualcomm AIHub product image with
vocabulary like person/phone/text; that second image is deferred to keep
the in-repo asset surface small. Tracked alongside the perplexity follow-up.
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
