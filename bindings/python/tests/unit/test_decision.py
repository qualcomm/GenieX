# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from ctypes import POINTER, c_float, c_void_p, cast, create_string_buffer

import pytest

from geniex._ffi._types import geniex_DecideInput, geniex_DecideOutput
from geniex.decision import DecideInput, DecideOption, DecideQuestion
from geniex.modeling import GenieXLLM, GenieXVLM


class FakeDecisionLibrary:
    def __init__(self):
        self.buffers = []
        self.freed = []
        self.calls = []
        self.geniex_free = self.freed.append

    def geniex_llm_decide(self, handle, input_ptr, output_ptr):
        return self._decide('llm', handle, input_ptr, output_ptr)

    def geniex_vlm_decide(self, handle, input_ptr, output_ptr):
        return self._decide('vlm', handle, input_ptr, output_ptr)

    def _decide(self, kind, handle, input_ptr, output_ptr):
        request = cast(input_ptr, POINTER(geniex_DecideInput)).contents
        question = request.questions[request.question_index]
        self.calls.append(
            {
                'kind': kind,
                'handle': handle,
                'state': request.state.decode(),
                'question_id': question.id.decode(),
                'question_type': question.question_type.decode(),
                'instructions': question.instructions.decode(),
                'options': [
                    (
                        question.options[index].key.decode(),
                        bool(question.options[index].key_is_bool),
                        question.options[index].description.decode() if question.options[index].description else None,
                    )
                    for index in range(question.option_count)
                ],
                'images': [request.image_paths[index].decode() for index in range(request.image_count)],
            }
        )
        logits = (c_float * 2)(1.25, 2.5)
        probabilities = (c_float * 2)(0.2, 0.8)
        decision_type = create_string_buffer(b'openjev')
        self.buffers.extend((logits, probabilities, decision_type))
        output = cast(output_ptr, POINTER(geniex_DecideOutput)).contents
        output.logits = cast(logits, POINTER(c_float))
        output.probabilities = cast(probabilities, POINTER(c_float))
        output.option_count = 2
        output.input_tokens = 17
        output.temperature = 0.75
        output.decision_type = cast(decision_type, c_void_p)
        return 0


@pytest.mark.parametrize(
    ('model_type', 'kind', 'images'),
    [(GenieXLLM, 'llm', []), (GenieXVLM, 'vlm', ['image.png'])],
)
def test_decide_calls_native_api_and_copies_owned_output(monkeypatch, model_type, kind, images):
    from geniex import modeling

    library = FakeDecisionLibrary()
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(model_type)
    model._handle = 42
    model._meta = None

    result = model.decide(
        DecideInput(
            state='customer context',
            questions=[
                DecideQuestion(
                    id='route',
                    question_type='choice',
                    instructions='Choose a route.',
                    options=[
                        DecideOption('self_service', description='Use documentation'),
                        DecideOption('agent', description='Talk to a person'),
                    ],
                )
            ],
            image_paths=images,
        )
    )

    assert result.logits == pytest.approx([1.25, 2.5])
    assert result.probabilities == pytest.approx([0.2, 0.8])
    assert result.input_tokens == 17
    assert result.temperature == pytest.approx(0.75)
    assert result.decision_type == 'openjev'
    assert library.calls == [
        {
            'kind': kind,
            'handle': 42,
            'state': 'customer context',
            'question_id': 'route',
            'question_type': 'choice',
            'instructions': 'Choose a route.',
            'options': [
                ('self_service', False, 'Use documentation'),
                ('agent', False, 'Talk to a person'),
            ],
            'images': images,
        }
    ]
    assert len(library.freed) == 3


def test_llm_decide_rejects_media_before_native_call(monkeypatch):
    from geniex import modeling

    library = FakeDecisionLibrary()
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(GenieXLLM)
    model._handle = 42
    model._meta = None
    request = DecideInput(
        state='state',
        questions=[DecideQuestion('q', 'choice', 'Choose.', [DecideOption('a')])],
        image_paths=['image.png'],
    )

    with pytest.raises(ValueError, match='cannot include media'):
        model.decide(request)
    assert library.calls == []
