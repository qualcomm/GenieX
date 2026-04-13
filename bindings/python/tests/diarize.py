"""
Pytest tests for Diarize functionality.
"""

import pytest

from geniex import Diarize
from tests.conftest import PROJECT_ROOT

DIARIZE_MODEL_CONFIGS = [
    {
        'plugin_id': 'npu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'qnn' / 'Pyannote-NPU' / 'weights-1-2.nexa',
        'model_name': 'pyannote',
    },
]


@pytest.fixture(params=DIARIZE_MODEL_CONFIGS)
def diarize_instance(request):
    """Create a Diarize instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return Diarize(
        model_path=str(config['model_path']),
        model_name=config.get('model_name'),
        plugin_id=config['plugin_id'],
    )


@pytest.fixture
def test_audio_path():
    """Return path to test audio file."""
    audio_path = PROJECT_ROOT / 'modelfiles' / 'assets' / 'conversation_16k.wav'
    if not audio_path.exists():
        pytest.skip(f'Test audio not found: {audio_path}')
    return str(audio_path)


def test_diarize_complete(diarize_instance, test_audio_path):
    """
    Complete Diarize test covering inference functionality.
    """
    result = diarize_instance.infer(test_audio_path)
    assert result is not None, 'Infer result should not be None'
    assert hasattr(result, 'segments'), 'Result should have segments attribute'
    assert isinstance(result.segments, list), 'Segments should be a list'
    if len(result.segments) > 0:
        assert hasattr(result.segments[0], 'speaker_label'), 'Segments should have speaker_label'
        assert hasattr(result.segments[0], 'start_time'), 'Segments should have start_time'
        assert hasattr(result.segments[0], 'end_time'), 'Segments should have end_time'
