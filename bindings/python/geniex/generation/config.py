# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Transformers-compatible generation parameters.

    Numeric sampler fields default to None (= bundle/plugin default); 0 is used as given.
    """

    max_new_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    grammar: str | None = None
    images: list[str] = field(default_factory=list)
    audios: list[str] = field(default_factory=list)
    stream: bool = False

    # Opt-in ring-buffer context eviction (qairt only).
    sliding_window: bool = False
    sliding_window_n_keep: int = 0  # 0 = plugin default (4)
