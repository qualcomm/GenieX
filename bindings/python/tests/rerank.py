"""
Pytest tests for Reranker functionality.
"""

from pathlib import Path

import pytest

from geniex import Reranker
from tests.conftest import PROJECT_ROOT

RERANK_MODEL_CONFIGS = [
    {
        'plugin_id': 'cpu_gpu',
        'model_path': PROJECT_ROOT / 'modelfiles' / 'llama_cpp' / 'bge-reranker-v2-m3-Q4_K_M.gguf',
        'model_name': 'bge-reranker-v2-m3',
    },
]


@pytest.fixture(params=RERANK_MODEL_CONFIGS)
def reranker_instance(request):
    """Create a Reranker instance for testing."""
    config = request.param
    if not config['model_path'].exists():
        pytest.skip(f'Model file not found: {config["model_path"]}')
    return Reranker(
        model_path=str(config['model_path']),
        model_name=config.get('model_name'),
        plugin_id=config['plugin_id'],
    )


def test_rerank_complete(reranker_instance):
    """
    Complete Reranker test covering rerank functionality.
    """
    query = 'What is machine learning?'
    documents = [
        'Machine learning is a subset of AI.',
        'Python is a programming language.',
        'Deep learning uses neural networks.',
    ]
    result = reranker_instance.rerank(query, documents)
    assert result is not None, 'Rerank result should not be None'
    assert hasattr(result, 'scores'), 'Result should have scores attribute'
    assert isinstance(result.scores, list), 'Scores should be a list'
    assert len(result.scores) == len(documents), 'Number of scores should match number of documents'
    assert all(isinstance(score, (int, float)) for score in result.scores), 'All scores should be numbers'
