# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Transformers-compatible generation parameters.

    Numeric sampler fields default to 0 (= defer to bundle/plugin default).
    """

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = 0
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None
    json_mode: bool = False
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    stream: bool = False

    # Opt-in ring-buffer context eviction (qairt only). See qcom-ai-hub/geniex#1197.
    sliding_window: bool = False
    sliding_window_n_keep: int = 0  # 0 = plugin default (4)
