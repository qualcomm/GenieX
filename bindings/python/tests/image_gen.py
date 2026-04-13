"""
Pytest tests for ImageGen functionality.
"""

from pathlib import Path

import pytest

from geniex import ImageGen
from tests.conftest import PROJECT_ROOT

IMAGE_GEN_MODEL_CONFIGS = [
    {
        'plugin_id': 'ort_dml',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'ort' / 'sdxl-fp16',
        'model_name': 'SDXL-FP16',
    },
]


@pytest.fixture(params=IMAGE_GEN_MODEL_CONFIGS)
def image_gen_instance(request):
    """Create an ImageGen instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model path not found: {config["model_path"]}')
    return ImageGen.from_(
        'runwayml/stable-diffusion-v1-5',
        plugin_id=config['plugin_id'],
    )


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory for test files."""
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    return str(output_dir)


def test_image_gen_complete(image_gen_instance, output_dir):
    """
    Complete ImageGen test covering text-to-image functionality.
    """
    prompt = 'a beautiful landscape'
    output_path = str(Path(output_dir) / 'generated_image.png')
    result = image_gen_instance.txt2img(prompt, output_path)
    assert result is not None, 'ImageGen result should not be None'
    assert hasattr(result, 'output_image_path'), 'Result should have output_image_path attribute'
    assert isinstance(result.output_image_path, str), 'Output image path should be a string'
    assert len(result.output_image_path) > 0, 'Output image path should not be empty'
    assert Path(result.output_image_path).exists(), 'Output image file should exist'
