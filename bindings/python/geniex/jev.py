# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""Typed, non-executing decisions backed by the SDK logits path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .decision import DecideInput, DecideOption, DecideOutput, DecideQuestion
from .modeling import GenieXLLM, GenieXVLM


@dataclass(frozen=True)
class Confidence:
    value: float
    provenance: str


@dataclass(frozen=True)
class ClassificationSpec:
    labels: list[str]
    instructions: str
    context: str = ''
    images: list[str] = field(default_factory=list)
    request_confidence: bool = False


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: Confidence | None = None


@dataclass(frozen=True)
class Option:
    id: str
    text: str


@dataclass(frozen=True)
class MultipleChoiceSpec:
    options: list[Option]
    instructions: str
    context: str = ''
    images: list[str] = field(default_factory=list)
    min_selections: int = 0
    max_selections: int = 0
    request_confidence: bool = False


@dataclass(frozen=True)
class MultipleChoiceResult:
    selected_ids: list[str]
    confidence: Confidence | None = None


@dataclass(frozen=True)
class TriageSpec:
    message: str
    images: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TriageResult:
    intent: str
    is_urgent: str
    frustration: str
    refund_requested: str
    churn_risk: str


_TRIAGE_INTENTS = ('refund', 'technical_help', 'billing_question', 'information', 'cancellation', 'other')
_PREDICATE_OPTIONS = (DecideOption('true'), DecideOption('false'))


class JevDecider:
    """Maps typed JEV requests to ``geniex_llm_decide`` or ``geniex_vlm_decide``."""

    def __init__(self, model: GenieXLLM | GenieXVLM) -> None:
        self._model = model

    def classify(self, spec: ClassificationSpec) -> ClassificationResult:
        labels = _validate_ids(spec.labels, 'classification labels')
        _require_instructions(spec.instructions, 'classification')
        question = DecideQuestion(
            id='classification',
            question_type='choice',
            instructions=spec.instructions,
            options=[DecideOption(label) for label in labels],
        )
        output = self._decide(spec.context, [question], 0, spec.images)
        probabilities, provenance = _probabilities(output, len(labels))
        selected = max(range(len(labels)), key=probabilities.__getitem__)
        confidence = Confidence(probabilities[selected], provenance) if spec.request_confidence else None
        return ClassificationResult(labels[selected], confidence)

    def choose(self, spec: MultipleChoiceSpec) -> MultipleChoiceResult:
        ids = _validate_ids([option.id for option in spec.options], 'multiple-choice option IDs')
        _require_instructions(spec.instructions, 'multiple-choice')
        count = _selection_count(spec.min_selections, spec.max_selections, len(ids))
        question = DecideQuestion(
            id='selection',
            question_type='choice',
            instructions=spec.instructions,
            options=[DecideOption(option.id, description=option.text) for option in spec.options],
        )
        output = self._decide(spec.context, [question], 0, spec.images)
        probabilities, provenance = _probabilities(output, len(ids))
        selected = sorted(range(len(ids)), key=probabilities.__getitem__, reverse=True)[:count]
        minimum_probability = min(probabilities[index] for index in selected)
        confidence = Confidence(minimum_probability, provenance) if spec.request_confidence else None
        return MultipleChoiceResult([ids[index] for index in selected], confidence)

    def triage(self, spec: TriageSpec) -> TriageResult:
        if not spec.message.strip():
            raise ValueError('triage message is required')
        questions = _triage_questions()
        values = []
        for index, question in enumerate(questions):
            output = self._decide(spec.message, questions, index, spec.images)
            probabilities, _ = _probabilities(output, len(question.options))
            values.append(question.options[max(range(len(question.options)), key=probabilities.__getitem__)].key)
        return TriageResult(
            intent=values[0],
            is_urgent=_yes_no(values[1]),
            frustration=values[2],
            refund_requested=_yes_no(values[3]),
            churn_risk=_yes_no(values[4]),
        )

    def _decide(
        self, state: str, questions: list[DecideQuestion], question_index: int, images: list[str]
    ) -> DecideOutput:
        if images and isinstance(self._model, GenieXLLM):
            raise ValueError('text structured decisions cannot include images')
        return self._model.decide(
            DecideInput(
                state=state,
                questions=questions,
                question_index=question_index,
                image_paths=list(images),
            )
        )


def _probabilities(output: DecideOutput, expected: int) -> tuple[list[float], str]:
    if output.probabilities:
        if len(output.probabilities) != expected:
            raise ValueError('SDK decision returned an unexpected probability count')
        return list(output.probabilities), 'sdk_calibrated'
    if len(output.logits) != expected:
        raise ValueError('SDK decision returned an unexpected logit count')
    if output.temperature <= 0 or not math.isfinite(output.temperature):
        raise ValueError('SDK decision returned an invalid temperature')
    peak = max(output.logits)
    weights = [math.exp((value - peak) / output.temperature) for value in output.logits]
    total = sum(weights)
    if not total or not math.isfinite(total):
        raise ValueError('SDK decision returned invalid logits')
    return [weight / total for weight in weights], 'logits_softmax'


def _validate_ids(values: list[str], label: str) -> list[str]:
    if not values:
        raise ValueError(f'{label} are required')
    if any(not value.strip() for value in values) or len(set(values)) != len(values):
        raise ValueError(f'{label} must be non-empty and unique')
    return values


def _require_instructions(value: str, label: str) -> None:
    if not value.strip():
        raise ValueError(f'{label} instructions are required')


def _selection_count(minimum: int, maximum: int, option_count: int) -> int:
    if minimum == maximum == 0:
        return 1
    if minimum != maximum:
        raise ValueError('native decisions require a fixed selection count')
    if minimum < 1 or minimum > option_count:
        raise ValueError('invalid selection count')
    return minimum


def _yes_no(value: str) -> str:
    return 'yes' if value == 'true' else 'no'


def _triage_questions() -> list[DecideQuestion]:
    return [
        DecideQuestion(
            'intent',
            'choice',
            'Classify the customer message intent.',
            [DecideOption(value) for value in _TRIAGE_INTENTS],
        ),
        DecideQuestion(
            'is_urgent',
            'predicate',
            'Does the message communicate time pressure or a deadline?',
            list(_PREDICATE_OPTIONS),
        ),
        DecideQuestion(
            'frustration',
            'score',
            'Score frustration from calm and neutral to very angry or strong language.',
            [
                DecideOption('0', description='calm and neutral'),
                DecideOption('1', description='concerned but civil'),
                DecideOption('2', description='clearly annoyed'),
                DecideOption('3', description='very angry or strong language'),
            ],
        ),
        DecideQuestion(
            'refund_requested',
            'predicate',
            'Does the customer ask for money back?',
            list(_PREDICATE_OPTIONS),
        ),
        DecideQuestion(
            'churn_risk',
            'predicate',
            'Does the message suggest leaving, cancelling, or using a competitor?',
            list(_PREDICATE_OPTIONS),
        ),
    ]
