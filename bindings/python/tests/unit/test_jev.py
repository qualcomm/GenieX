# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from geniex.decision import DecideOutput
from geniex.jev import ClassificationSpec, JevDecider, MultipleChoiceSpec, Option, TriageSpec
from geniex.modeling import GenieXLLM, GenieXVLM


class FakeVLM(GenieXVLM):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


class FakeLLM(GenieXLLM):
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return self.outputs.pop(0)


def test_classify_uses_native_probabilities_and_preserves_media():
    model = FakeVLM([DecideOutput(probabilities=[0.25, 0.75], temperature=0.5, decision_type='lev')])

    result = JevDecider(model).classify(
        ClassificationSpec(
            labels=['cat', 'dog'],
            instructions='Classify the animal.',
            context='A friendly pet.',
            images=['pet.png'],
            request_confidence=True,
        )
    )

    assert result.label == 'dog'
    assert result.confidence is not None
    assert result.confidence.value == pytest.approx(0.75)
    assert result.confidence.provenance == 'sdk_calibrated'
    request = model.requests[0]
    assert request.state == 'A friendly pet.'
    assert request.image_paths == ['pet.png']
    assert request.question_index == 0
    assert [(option.key, option.description) for option in request.questions[0].options] == [
        ('cat', None),
        ('dog', None),
    ]


def test_classify_softmaxes_logits_with_sdk_temperature():
    model = FakeLLM([DecideOutput(logits=[0.0, 2.0], temperature=2.0, decision_type='openjev')])

    result = JevDecider(model).classify(
        ClassificationSpec(labels=['a', 'b'], instructions='Choose.', request_confidence=True)
    )

    assert result.label == 'b'
    assert result.confidence is not None
    assert result.confidence.value == pytest.approx(0.7310586)
    assert result.confidence.provenance == 'logits_softmax'


def test_choose_returns_fixed_top_k_and_rejects_ranged_selection():
    model = FakeLLM([DecideOutput(logits=[1.0, 4.0, 3.0], temperature=1.0, decision_type='generic')])
    spec = MultipleChoiceSpec(
        options=[Option('a', 'Alpha'), Option('b', 'Beta'), Option('c', 'Gamma')],
        instructions='Choose two.',
        min_selections=2,
        max_selections=2,
        request_confidence=True,
    )

    result = JevDecider(model).choose(spec)

    assert result.selected_ids == ['b', 'c']
    assert result.confidence is not None
    assert result.confidence.value == pytest.approx(0.25949646)
    assert result.confidence.provenance == 'logits_softmax'
    assert [(option.key, option.description) for option in model.requests[0].questions[0].options] == [
        ('a', 'Alpha'),
        ('b', 'Beta'),
        ('c', 'Gamma'),
    ]

    with pytest.raises(ValueError, match='fixed selection count'):
        JevDecider(model).choose(
            MultipleChoiceSpec(options=spec.options, instructions='Choose.', min_selections=1, max_selections=2)
        )


def test_triage_runs_each_native_question_and_maps_predicates():
    model = FakeVLM(
        [
            DecideOutput(probabilities=[0.8, 0.05, 0.04, 0.03, 0.04, 0.04], decision_type='openjev'),
            DecideOutput(probabilities=[0.9, 0.1], decision_type='openjev'),
            DecideOutput(probabilities=[0.05, 0.15, 0.7, 0.1], decision_type='openjev'),
            DecideOutput(probabilities=[0.85, 0.15], decision_type='openjev'),
            DecideOutput(probabilities=[0.2, 0.8], decision_type='openjev'),
        ]
    )

    result = JevDecider(model).triage(TriageSpec(message='I was charged twice and will leave.', images=['bill.png']))

    assert result.intent == 'refund'
    assert result.is_urgent == 'yes'
    assert result.frustration == '2'
    assert result.refund_requested == 'yes'
    assert result.churn_risk == 'no'
    assert [request.question_index for request in model.requests] == [0, 1, 2, 3, 4]
    assert all(request.image_paths == ['bill.png'] for request in model.requests)
    assert [question.question_type for question in model.requests[0].questions] == [
        'choice',
        'predicate',
        'score',
        'predicate',
        'predicate',
    ]


def test_llm_rejects_images_before_native_decision():
    model = FakeLLM([DecideOutput(probabilities=[1.0])])

    with pytest.raises(ValueError, match='cannot include images'):
        JevDecider(model).classify(ClassificationSpec(labels=['yes'], instructions='Answer.', images=['image.png']))
    assert model.requests == []
