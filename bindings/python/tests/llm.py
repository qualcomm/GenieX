"""
Pytest tests for LLM (Large Language Model) functionality.
"""

import pytest

from geniex import LLM, GenerationConfig, LlmChatMessage
from tests.conftest import PROJECT_ROOT

LLM_MODEL_CONFIGS = [
    {
        'plugin_id': 'cpu_gpu',
        'model_path': (PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'Qwen3-0.6B-Q8_0.gguf'),
        'model_name': None,
    },
]


@pytest.fixture(params=LLM_MODEL_CONFIGS)
def llm_instance(request):
    """
    Create an LLM instance for testing.
    """
    config = request.param
    model_path = str(config['model_path'])
    model_name = config.get('model_name', model_path)

    return LLM(
        model_path=model_path,
        model_name=model_name,
        plugin_id=config['plugin_id'],
    )


def test_llm_complete(llm_instance):
    """
    Complete LLM test covering all functionality:
    1. Chat template application
    2. Stream generation
    3. Basic generation (non-streaming)
    This test matches the functionality of the original script-based test.
    """
    formatted_prompt = llm_instance.apply_chat_template(
        [
            LlmChatMessage(role='system', content='You are a helpful assistant.'),
            LlmChatMessage(role='user', content='Hello, how are you?'),
        ]
    )
    assert isinstance(formatted_prompt, str), 'Formatted prompt should be a string'
    assert len(formatted_prompt) > 0, 'Formatted prompt should not be empty'

    # Test: Generate stream
    tokens = []
    for token in llm_instance.generate_stream(formatted_prompt, GenerationConfig(max_tokens=512)):
        tokens.append(token)
    print(f'Tokens: {tokens}')
    # Verify stream generation output
    assert len(tokens) > 0, 'Expected at least one token in the generated output'
    assert all(isinstance(token, str) for token in tokens), 'All tokens should be strings'
    full_text = ''.join(tokens)
    assert len(full_text) > 0, 'Generated text should not be empty'

    # Test: Basic generation (non-streaming)
    prompt = 'Hello, how are you?'
    result = llm_instance.generate(prompt, GenerationConfig(max_tokens=100))
    assert result is not None, 'Generate result should not be None'
    assert hasattr(result, 'full_text'), 'Result should have full_text attribute'
    assert isinstance(result.full_text, str), 'Result full_text should be a string'
    assert len(result.full_text) > 0, 'Generated text should not be empty'
    print(f'Full text: {full_text}')
