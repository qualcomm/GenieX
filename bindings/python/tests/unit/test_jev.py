# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from geniex.generation import GenerateOutput, GenerationConfig
from geniex.jev import (
    ClassificationSpec,
    JevDecider,
    MultipleChoiceSpec,
    Option,
    TriageSpec,
    _choice_grammar,
    _classification_grammar,
    decode_strict,
)
from geniex.modeling import GenieXLLM, GenieXVLM


class FakeVLM(GenieXVLM):
    def __init__(self, response: str, backend: str = 'llama_cpp'):
        self.response = response
        self._meta = {'backend': backend}
        self.calls = []
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'formatted'

    def generate(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return GenerateOutput(text=self.response)


class FakeLLM(GenieXLLM):
    def __init__(self, response: str):
        self.response = response
        self._meta = {'backend': 'llama_cpp'}
        self.tokenizer = self

    def apply_chat_template(self, messages, **kwargs):
        return 'formatted'

    def generate(self, prompt, **kwargs):
        return GenerateOutput(text=self.response)


def test_decode_strict_rejects_duplicates_and_trailing_text():
    assert decode_strict('```json\n{"answer":"yes"}\n```') == {'answer': 'yes'}
    for payload in ('{"answer":"yes","answer":"no"}', '{"answer":"yes"} prose', '["yes"]'):
        with pytest.raises(ValueError):
            decode_strict(payload)


def test_classify_forwards_generation_config_and_llama_grammar():
    model = FakeVLM('{"label":"cat","confidence":0.75}')
    decider = JevDecider(model, GenerationConfig(top_k=4, stop=['</s>'], sliding_window=True, sliding_window_n_keep=8))
    result = decider.classify(
        ClassificationSpec(
            labels=['cat', 'dog'],
            instructions='Classify.',
            images=['image.png'],
            request_confidence=True,
        )
    )
    assert result.label == 'cat'
    assert result.confidence and result.confidence.value == 0.75
    _, kwargs = model.calls[0]
    assert kwargs['top_k'] == 4
    assert kwargs['stop'] == ['</s>']
    assert kwargs['sliding_window'] is True
    assert kwargs['sliding_window_n_keep'] == 8
    assert 'cat' in kwargs['grammar']
    assert kwargs['images'] == ['image.png']


def test_qairt_omits_grammar_but_validates_result():
    model = FakeVLM('{"selected_ids":["a"]}', backend='qairt')
    result = JevDecider(model).choose(
        MultipleChoiceSpec(
            options=[Option('a', 'First'), Option('b', 'Second')],
            instructions='Choose.',
        )
    )
    assert result.selected_ids == ['a']
    assert model.calls[0][1]['grammar'] is None


def test_direct_model_without_metadata_uses_unconstrained_json():
    model = FakeVLM('{"label":"cat"}')
    model._meta = None

    assert JevDecider(model).classify(ClassificationSpec(labels=['cat'], instructions='Classify.')).label == 'cat'
    assert model.calls[0][1]['grammar'] is None


def test_jev_replaces_caller_grammar_with_host_owned_grammar():
    model = FakeVLM('{"label":"cat"}')
    config = GenerationConfig(grammar='root ::= unsafe')
    JevDecider(model, config).classify(ClassificationSpec(labels=['cat'], instructions='Classify.'))
    assert model.calls[0][1]['grammar'] != 'root ::= unsafe'
    assert 'cat' in model.calls[0][1]['grammar']


def test_generated_grammar_matches_jev_whitespace_and_cardinality_contract():
    classification = _classification_grammar(['cat'], False)
    choice = _choice_grammar(['a', 'b'], 1, 2, False)
    assert 'ws ::= [ \\t\\n\\r]*' in classification
    assert '"\\"cat\\""' in classification
    assert 'selections ::= id | id ws "," ws id' in choice
    assert 'number ::= "0" ("." [0-9]+)? | "1" ("." "0"+)?' in classification


def test_llm_rejects_images_and_triage_validates_categories():
    model = FakeLLM('{"intent":"refund","is_urgent":"no","frustration":"2","refund_requested":"yes","churn_risk":"no"}')
    decider = JevDecider(model)
    result = decider.triage(TriageSpec(message='I was charged twice and want a refund'))
    assert result.intent == 'refund'
    with pytest.raises(ValueError, match='cannot include images'):
        decider.classify(ClassificationSpec(labels=['yes'], instructions='Answer.', images=['image.png']))
