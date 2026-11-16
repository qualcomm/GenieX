# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from ctypes import POINTER, c_int32, c_void_p, cast, create_string_buffer, string_at

import pytest

from geniex._ffi._types import (
    geniex_ChatTemplateGrammar,
    geniex_GrammarTrigger,
    geniex_LlmApplyChatTemplateInput,
    geniex_LlmApplyChatTemplateOutput,
    geniex_VlmApplyChatTemplateInput,
    geniex_VlmApplyChatTemplateOutput,
)
from geniex.chat_template import ChatTemplateGrammar, ChatTemplateOutput, GrammarTrigger
from geniex.modeling import (
    GenieXLLM,
    GenieXVLM,
    _build_sampler,
    _copy_chat_template_grammar,
)
from geniex.tokenizer import ModelTokenizer


def _make_grammar_buffers():
    grammar_text = create_string_buffer(b'root ::= \\"ok\\"')
    generation_prompt = create_string_buffer(b'<|start|>assistant')
    trigger_value = create_string_buffer(b'<|channel|>analysis to')
    triggers = (geniex_GrammarTrigger * 1)()
    triggers[0].trigger_type = 2
    triggers[0].value = cast(trigger_value, c_void_p)
    triggers[0].token = 0
    preserved = (c_int32 * 2)(200005, 200012)
    grammar = geniex_ChatTemplateGrammar(
        grammar_type=3,
        grammar=cast(grammar_text, c_void_p),
        lazy=True,
        generation_prompt=cast(generation_prompt, c_void_p),
        triggers=triggers,
        trigger_count=1,
        preserved_tokens=preserved,
        preserved_token_count=2,
    )
    return grammar, (grammar_text, generation_prompt, trigger_value, triggers, preserved)


def test_copy_chat_template_grammar_owns_all_fields():
    grammar, keepalive = _make_grammar_buffers()
    copied = _copy_chat_template_grammar(grammar)
    del keepalive

    assert copied.grammar_type == 3
    assert copied.grammar == 'root ::= \\"ok\\"'
    assert copied.lazy is True
    assert copied.generation_prompt == '<|start|>assistant'
    assert copied.triggers == (GrammarTrigger(2, '<|channel|>analysis to', 0),)
    assert copied.preserved_tokens == (200005, 200012)


def test_build_sampler_retains_chat_grammar_buffers():
    metadata = ChatTemplateGrammar(
        grammar_type=3,
        grammar='root ::= "{}"',
        lazy=True,
        generation_prompt='prompt',
        triggers=(GrammarTrigger(2, 'to=functions', 0),),
        preserved_tokens=(200005,),
    )
    sampler = _build_sampler(0, 0, 0, 0, 0, 0, 0, 0, None, metadata)
    grammar = sampler.chat_template_grammar

    assert grammar.grammar_type == 3
    assert grammar.lazy
    assert string_at(grammar.grammar) == b'root ::= "{}"'
    assert string_at(grammar.generation_prompt) == b'prompt'
    assert grammar.trigger_count == 1
    assert string_at(grammar.triggers[0].value) == b'to=functions'
    assert grammar.preserved_token_count == 1
    assert grammar.preserved_tokens[0] == 200005
    assert sampler._chat_template_grammar_keepalive


def test_tokenizer_preserves_default_string_and_opt_in_result():
    class FakeModel:
        supports_thinking = True

        def _apply_chat_template(self, **kwargs):
            self.kwargs = kwargs
            result = ChatTemplateOutput('prompt', ChatTemplateGrammar())
            return result if kwargs['include_grammar'] else result.formatted_text

    model = FakeModel()
    tokenizer = ModelTokenizer(model)
    messages = [{'role': 'user', 'content': 'hello'}]

    assert tokenizer.apply_chat_template(messages) == 'prompt'
    assert model.kwargs['include_grammar'] is False
    result = tokenizer.apply_chat_template(messages, tools=[], include_grammar=True)
    assert isinstance(result, ChatTemplateOutput)
    assert result.formatted_text == 'prompt'
    assert model.kwargs['tools'] == '[]'
    assert model.kwargs['parallel_tool_calls'] is True
    tokenizer.apply_chat_template(messages, parallel_tool_calls=False)
    assert model.kwargs['parallel_tool_calls'] is False


@pytest.mark.parametrize('model_type', [GenieXLLM, GenieXVLM])
def test_generate_forwards_template_grammar_to_sampler(monkeypatch, model_type):
    from geniex import modeling

    metadata = ChatTemplateGrammar(
        grammar_type=3,
        grammar='root ::= "{}"',
        lazy=True,
        generation_prompt='prompt',
        triggers=(GrammarTrigger(2, 'to=functions', 0),),
        preserved_tokens=(200005,),
    )
    model = object.__new__(model_type)
    model._meta = None
    if model_type is GenieXVLM:
        model._last_template_has_image = False
        model._last_template_has_audio = False
    monkeypatch.setattr(modeling, '_build_gen_config', lambda *args: (None, None, None, None))
    captured = {}
    model._generate_blocking = lambda prompt, cfg, sampler, *_keep, **_kwargs: captured.setdefault('sampler', sampler)

    model.generate('prompt', chat_template_grammar=metadata)

    assert captured['sampler'].chat_template_grammar.grammar_type == 3
    assert string_at(captured['sampler'].chat_template_grammar.triggers[0].value) == b'to=functions'


def test_default_sampler_has_no_template_grammar():
    sampler = _build_sampler(0, 0, 0, 0, 0, 0, 0, 0, None)
    assert sampler.chat_template_grammar.grammar_type == 0
    assert not sampler.chat_template_grammar.grammar
    assert not sampler.chat_template_grammar.triggers


class _FakeLibrary:
    def __init__(self, output_type):
        self.output_type = output_type
        self.input_type = (
            geniex_LlmApplyChatTemplateInput
            if output_type is geniex_LlmApplyChatTemplateOutput
            else geniex_VlmApplyChatTemplateInput
        )
        self.buffers = []
        self.freed = []
        self.geniex_free = self.freed.append
        self.apply_name = (
            'geniex_llm_apply_chat_template'
            if output_type is geniex_LlmApplyChatTemplateOutput
            else 'geniex_vlm_apply_chat_template'
        )
        setattr(self, self.apply_name, self._apply)

    def _apply(self, _handle, _input, output_ptr):
        inp = cast(_input, POINTER(self.input_type)).contents
        self.parallel_tool_calls = inp.parallel_tool_calls
        self.reasoning_contents = [inp.messages[i].reasoning_content for i in range(inp.message_count)]
        output = cast(output_ptr, POINTER(self.output_type)).contents
        text = create_string_buffer(b'formatted prompt')
        grammar, metadata_buffers = _make_grammar_buffers()
        self.buffers.extend([text, *metadata_buffers])
        output.formatted_text = cast(text, c_void_p)
        output.chat_template_grammar = grammar
        return 0


@pytest.mark.parametrize(
    ('model_type', 'output_type', 'messages'),
    [
        (GenieXLLM, geniex_LlmApplyChatTemplateOutput, [{'role': 'user', 'content': 'hello'}]),
        (GenieXVLM, geniex_VlmApplyChatTemplateOutput, [{'role': 'user', 'content': 'hello'}]),
    ],
)
def test_apply_template_includes_owned_grammar_and_frees_c_output(monkeypatch, model_type, output_type, messages):
    from geniex import modeling

    library = _FakeLibrary(output_type)
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(model_type)
    model._handle = 1
    model._meta = None

    result = model._apply_chat_template(messages, True, True, None, include_grammar=True)
    model._handle = None

    assert isinstance(result, ChatTemplateOutput)
    assert result.formatted_text == 'formatted prompt'
    assert result.chat_template_grammar.grammar == 'root ::= \\"ok\\"'
    assert result.chat_template_grammar.triggers[0].value == '<|channel|>analysis to'
    assert len(library.freed) == 6


def test_apply_template_error_still_frees_partial_output(monkeypatch):
    from geniex import modeling

    library = _FakeLibrary(geniex_LlmApplyChatTemplateOutput)
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    monkeypatch.setattr(modeling, '_check', lambda _result: (_ for _ in ()).throw(RuntimeError('SDK failure')))
    model = object.__new__(GenieXLLM)
    model._handle = 1
    model._meta = None

    with pytest.raises(RuntimeError, match='SDK failure'):
        model._apply_chat_template([{'role': 'user', 'content': 'hello'}], True, True, None, include_grammar=True)
    model._handle = None
    assert len(library.freed) == 6


@pytest.mark.parametrize(
    ('model_type', 'output_type'),
    [(GenieXLLM, geniex_LlmApplyChatTemplateOutput), (GenieXVLM, geniex_VlmApplyChatTemplateOutput)],
)
@pytest.mark.parametrize('kwargs', [{}, {'parallel_tool_calls': True}, {'parallel_tool_calls': False}])
def test_parallel_tool_calls_reaches_ffi(monkeypatch, model_type, output_type, kwargs):
    from geniex import modeling

    library = _FakeLibrary(output_type)
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(model_type)
    model._handle = 1
    model._meta = None
    model._apply_chat_template([{'role': 'user', 'content': 'hello'}], True, True, None, **kwargs)
    model._handle = None
    assert library.parallel_tool_calls is kwargs.get('parallel_tool_calls', True)


@pytest.mark.parametrize(
    ('model_type', 'output_type'),
    [(GenieXLLM, geniex_LlmApplyChatTemplateOutput), (GenieXVLM, geniex_VlmApplyChatTemplateOutput)],
)
def test_reasoning_content_reaches_ffi(monkeypatch, model_type, output_type):
    from geniex import modeling

    library = _FakeLibrary(output_type)
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(model_type)
    model._handle = 1
    model._meta = None
    messages = [
        {'role': 'user', 'content': 'weather?'},
        {'role': 'assistant', 'content': 'sunny', 'reasoning_content': 'check the forecast'},
    ]
    model._apply_chat_template(messages, True, True, None)
    model._handle = None
    assert library.reasoning_contents == [None, b'check the forecast']


def test_apply_template_default_still_returns_text_and_frees_grammar(monkeypatch):
    from geniex import modeling

    library = _FakeLibrary(geniex_LlmApplyChatTemplateOutput)
    monkeypatch.setattr(modeling, 'load_library', lambda: library)
    model = object.__new__(GenieXLLM)
    model._handle = 1
    model._meta = None

    result = model._apply_chat_template([{'role': 'user', 'content': 'hello'}], True, True, None)
    model._handle = None

    assert result == 'formatted prompt'
    assert len(library.freed) == 6
