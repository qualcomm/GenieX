"""
Pytest tests for TTS (Text-to-Speech) functionality.
"""

from pathlib import Path

import pytest

from geniex import TTS
from tests.conftest import PROJECT_ROOT

TTS_MODEL_CONFIGS = [
    {
        'plugin_id': 'cpu_gpu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'Qwen3-0.6B-Q8_0.gguf',
        'model_name': 'Qwen3-0.6B-Q8_0',
    },
]


@pytest.fixture(params=TTS_MODEL_CONFIGS)
def tts_instance(request):
    """Create a TTS instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return TTS(
        model_path=str(config['model_path']),
        plugin_id=config['plugin_id'],
    )


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory for test files."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    return str(output_dir)


def test_tts_complete(tts_instance, output_dir):
    """
    Complete TTS test covering synthesize functionality.
    """
    text = 'Hello world!'
    output_path = str(Path(output_dir) / 'tts_test.wav')
    result = tts_instance.synthesize(text, output_path)
    assert result is not None, 'Synthesize result should not be None'
    assert hasattr(result, 'audio_path'), 'Result should have audio_path attribute'
    assert isinstance(result.audio_path, str), 'Audio path should be a string'
    assert len(result.audio_path) > 0, 'Audio path should not be empty'
    assert Path(result.audio_path).exists(), 'Output audio file should exist'
