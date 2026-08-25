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

Thinking is disabled for the keyword cells. With it on, Qwen3 spends most of
`LLM_QUALITY_MAX_NEW_TOKENS` on a reasoning trace, the answer gets truncated,
and the binding strips the trace — leaving a stub that looks like a missing
keyword (qcom-ai-hub/geniex#1460).

One intentional delta vs. upstream: VLM only ships the dog photo + first
keyword set. Upstream also iterates a Qualcomm AIHub product image with
vocabulary like person/phone/text; that second image is deferred to keep
the in-repo asset surface small. Tracked alongside the perplexity follow-up.
"""

from __future__ import annotations

import math

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


PARITY_INPUT_IDS: list[int] = [
    1,
    2,
    3,
    5,
    8,
    13,
    21,
    34,
    55,
    89,
    144,
    233,
    377,
    610,
    987,
    1597,
    2584,
    4181,
    6765,
    10946,
    17711,
    28657,
    46368,
    75025,
    121393,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
]
PARITY_TOP1_MIN = 0.70
PARITY_KL_MAX = 0.35
PARITY_QAIRT_KL_MAX = 1e-4


def parity_softmax(row: list[float]) -> list[float]:
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def parity_kl_divergence(p_logits: list[float], q_logits: list[float]) -> float:
    p = parity_softmax(p_logits)
    q = parity_softmax(q_logits)
    return sum(pi * (math.log(pi) - math.log(max(qi, 1e-30))) for pi, qi in zip(p, q) if pi > 0)


def parity_top1_agreement(
    ref_rows: list[list[tuple[int, float]]],
    cand_rows: list[list[tuple[int, float]]],
) -> float:
    hits = sum(1 for a, b in zip(ref_rows, cand_rows) if a[0][0] == b[0][0])
    return hits / len(ref_rows)
