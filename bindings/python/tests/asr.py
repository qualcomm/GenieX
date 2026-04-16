"""
Pytest tests for ASR (Automatic Speech Recognition) functionality.
"""

import pytest

from geniex import ASR
from tests.conftest import PROJECT_ROOT

ASR_MODEL_CONFIGS = [
    {
        'plugin_id': 'npu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'qnn' / 'parakeet-tdt-0.6b-v3-npu' / 'weights-1-5.nexa',
        'model_name': 'parakeet',
    },
]


@pytest.fixture(params=ASR_MODEL_CONFIGS)
def asr_instance(request):
    """Create an ASR instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return ASR(
        model_path=str(config['model_path']),
        model_name=config.get('model_name'),
        plugin_id=config['plugin_id'],
        language='en',
    )


@pytest.fixture
def test_audio_path():
    """Return path to test audio file."""
    audio_path = PROJECT_ROOT / 'modelfiles' / 'assets' / 'OSR_us_000_0010_16k.wav'
    if not audio_path.exists():
        pytest.skip(f'Test audio not found: {audio_path}')
    return str(audio_path)


def test_asr_complete(asr_instance, test_audio_path):
    """
    Complete ASR test covering transcription functionality.
    """
    result = asr_instance.transcribe(test_audio_path)
    assert result is not None, 'Transcribe result should not be None'
    assert hasattr(result, 'transcript'), 'Result should have transcript attribute'
    assert isinstance(result.transcript, str), 'Transcript should be a string'
    assert len(result.transcript) > 0, 'Transcript should not be empty'
