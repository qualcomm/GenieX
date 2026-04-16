"""
Pytest tests for CV (Computer Vision) functionality.
"""

import pytest

from geniex import CV
from tests.conftest import PROJECT_ROOT

CV_MODEL_CONFIGS = [
    {
        'plugin_id': 'npu',
        'rec_model_path': PROJECT_ROOT / 'modelfiles' / 'qnn' / 'paddleocr-npu' / 'attachments-1-1.nexa',
        'model_name': 'paddleocr',
    },
]


@pytest.fixture(params=CV_MODEL_CONFIGS)
def cv_instance(request):
    """Create a CV instance for testing."""
    config = request.param
    return CV(
        model_name=config['model_name'],
        rec_model_path=str(config['rec_model_path']),
        plugin_id=config['plugin_id'],
    )


@pytest.fixture
def test_image_path():
    """Return path to test image."""
    image_path = PROJECT_ROOT / 'modelfiles' / 'assets' / 'test_image.png'
    if not image_path.exists():
        pytest.skip(f'Test image not found: {image_path}')
    return str(image_path)


def test_cv_complete(cv_instance, test_image_path):
    """
    Complete CV test covering inference functionality.
    """
    result = cv_instance.infer(test_image_path)
    assert result is not None, 'Infer result should not be None'
    assert hasattr(result, 'results'), 'Result should have results attribute'
    assert isinstance(result.results, list), 'Results should be a list'
    # Results may be empty, so we just check the structure
    if len(result.results) > 0:
        assert hasattr(result.results[0], 'text'), 'Result items should have text attribute'
