"""
Pytest tests for VLM (Vision-Language Model) functionality.
"""

import pytest

from geniex import VLM, GenerationConfig, VlmChatMessage, VlmContent
from tests.conftest import PROJECT_ROOT

VLM_MODEL_CONFIGS = [
    {
        'plugin_id': 'cpu_gpu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'SmolVLM-256M-Instruct-Q8_0.gguf',
        'mmproj_path': PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
        'model_name': 'SmolVLM-256M-Q8',
    },
]


@pytest.fixture(params=VLM_MODEL_CONFIGS)
def vlm_instance(request):
    """Create a VLM instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return VLM(
        model_path=str(config['model_path']),
        mmproj_path=str(config.get('mmproj_path', config['model_path'])),
        model_name=config.get('model_name'),
        plugin_id=config['plugin_id'],
    )


@pytest.fixture
def test_image_path():
    """Return path to test image."""
    image_path = PROJECT_ROOT / 'modelfiles' / 'assets' / 'test_image.png'
    if not image_path.exists():
        pytest.skip(f'Test image not found: {image_path}')
    return str(image_path)


def test_vlm_complete(vlm_instance, test_image_path):
    """
    Complete VLM test covering all functionality:
    1. Apply chat template with image
    2. Stream generation
    """
    formatted_prompt = vlm_instance.apply_chat_template(
        [
            VlmChatMessage(
                role='user',
                contents=[
                    VlmContent(type='text', text='Describe the image.'),
                    VlmContent(type='image', text=test_image_path),
                ],
            ),
        ]
    )
    assert isinstance(formatted_prompt, str), 'Formatted prompt should be a string'
    assert len(formatted_prompt) > 0, 'Formatted prompt should not be empty'

    # Test: Generate stream
    tokens = []
    for token in vlm_instance.generate_stream(
        formatted_prompt,
        GenerationConfig(max_tokens=128, image_paths=[test_image_path]),
    ):
        tokens.append(token)
    # Verify stream generation output
    assert len(tokens) > 0, 'Expected at least one token in the generated output'
    assert all(isinstance(token, str) for token in tokens), 'All tokens should be strings'
    full_text = ''.join(tokens)
    assert len(full_text) > 0, 'Generated text should not be empty'
    print(f'Full text: {full_text}')

    # Test: Generate
    result = vlm_instance.generate(
        formatted_prompt,
        GenerationConfig(max_tokens=128, image_paths=[test_image_path]),
    )
    assert len(result.full_text) > 0, 'Generated text should not be empty'
    print(f'Full text: {result.full_text}')
