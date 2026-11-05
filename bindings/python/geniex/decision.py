# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Typed inputs and outputs for SDK logits-based decisions."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecideOption:
    key: str
    key_is_bool: bool = False
    description: str | None = None


@dataclass(frozen=True)
class DecideQuestion:
    id: str
    question_type: str
    instructions: str
    options: list[DecideOption]


@dataclass(frozen=True)
class DecideInput:
    state: str
    questions: list[DecideQuestion]
    question_index: int = 0
    image_paths: list[str] = field(default_factory=list)
    audio_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DecideOutput:
    logits: list[float] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    input_tokens: int = 0
    temperature: float = 1.0
    decision_type: str = ''
