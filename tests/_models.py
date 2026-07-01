# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Test-model identifiers for the SDK pytest matrix.

One model per (plugin, modality) cell — kept here (not in conftest) so test
modules can import the names directly without needing a package layout.

Invariant: per modality, the llama_cpp and QAIRT cells point at the same
model — same family, same size, same training. AI Hub publishes nothing
smaller than Qwen3-4B (LLM) and Qwen2.5-VL-7B (VLM) in the Qwen family,
so the GGUF side is lifted up to match rather than QAIRT being scaled
down. The LLM is Qwen3-4B base (not the Instruct-2507 variant): the
Instruct tune emits a long ``<think>...</think>`` preamble before the
answer, which on a 256-token budget pushes the keyword off the end of
the completion and turns ``test_quality_keywords`` into a thinking-budget
test instead of a backend-quality test. With the cells aligned, a
keyword-quality divergence between the two plugins traces to backend /
quantization rather than model identity.

QAIRT entries honour ``GENIEX_QAIRT_MODEL`` / ``GENIEX_QAIRT_VLM_MODEL`` so
operators can swap in alternate models without editing the suite.
"""

from __future__ import annotations

import os

LLAMA_CPP_LLM_MODEL = 'unsloth/Qwen3-4B-GGUF'
LLAMA_CPP_LLM_PRECISION = 'Q4_0'
LLAMA_CPP_VLM_MODEL = 'unsloth/Qwen2.5-VL-7B-Instruct-GGUF'

QAIRT_LLM_MODEL = os.environ.get('GENIEX_QAIRT_MODEL', 'qualcomm/Qwen3-4B')
QAIRT_VLM_MODEL = os.environ.get('GENIEX_QAIRT_VLM_MODEL', 'qualcomm/Qwen2.5-VL-7B-Instruct')
