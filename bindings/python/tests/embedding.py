"""
Pytest tests for Embedding functionality.
"""

import pytest

from geniex import Embedder
from tests.conftest import PROJECT_ROOT

EMBEDDING_MODEL_CONFIGS = [
    {
        'plugin_id': 'cpu_gpu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'jina-embeddings-v2-small-en-Q4_K_M.gguf',
        'model_name': 'jina-v2-small-Q4',
    },
]


@pytest.fixture(params=EMBEDDING_MODEL_CONFIGS)
def embedder_instance(request):
    """Create an Embedder instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return Embedder(
        model_path=str(config['model_path']),
        model_name=config.get('model_name'),
        plugin_id=config['plugin_id'],
    )


def test_embedding_complete(embedder_instance):
    """
    Complete Embedding test covering embed functionality.
    """
    texts = ['Hello world', 'Machine learning']
    result = embedder_instance.embed(texts)
    assert result is not None, 'Embed result should not be None'
    assert hasattr(result, 'embeddings'), 'Result should have embeddings attribute'
    assert isinstance(result.embeddings, list), 'Embeddings should be a list'
    assert len(result.embeddings) == len(texts), 'Number of embeddings should match input texts'
    assert all(isinstance(emb, list) for emb in result.embeddings), 'Each embedding should be a list'
    assert all(len(emb) > 0 for emb in result.embeddings), 'Each embedding should have dimensions'
