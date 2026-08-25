# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Test-model identifiers, loaded from ``tests/models.json``."""

from __future__ import annotations

import json
import os
from pathlib import Path

_MANIFEST_PATH = Path(__file__).with_name('models.json')


def _pick(entry: dict, field: str = 'id') -> str | None:
    if field == 'id':
        env = entry.get('env_override')
        if env:
            override = os.environ.get(env)
            if override:
                return override
    return entry.get(field)


with _MANIFEST_PATH.open(encoding='utf-8') as _f:
    _MODELS = json.load(_f)['models']

LLAMA_CPP_LLM_MODEL = _pick(_MODELS['llama_cpp_llm'])
LLAMA_CPP_LLM_PRECISION = _pick(_MODELS['llama_cpp_llm'], 'precision')
LLAMA_CPP_VLM_MODEL = _pick(_MODELS['llama_cpp_vlm'])
LLAMA_CPP_VLM_PRECISION = _pick(_MODELS['llama_cpp_vlm'], 'precision')
LLAMA_CPP_MTP_TARGET_MODEL = _pick(_MODELS['llama_cpp_mtp_target'])
LLAMA_CPP_MTP_TARGET_PRECISION = _pick(_MODELS['llama_cpp_mtp_target'], 'precision')
LLAMA_CPP_MTP_DRAFT_MODEL = _pick(_MODELS['llama_cpp_mtp_draft'])
LLAMA_CPP_MTP_DRAFT_PRECISION = _pick(_MODELS['llama_cpp_mtp_draft'], 'precision')
QAIRT_LLM_MODEL = _pick(_MODELS['qairt_llm'])
QAIRT_VLM_MODEL = _pick(_MODELS['qairt_vlm'])
