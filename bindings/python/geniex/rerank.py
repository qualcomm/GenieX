"""
Rerank wrapper for geniex.
Provides high-level Python API for reranking operations.
"""

import ctypes
import logging
from typing import List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import check_error
from .geniex_sdk.llm import ml_ModelConfig as CModelConfig
from .geniex_sdk.rerank import (
    ml_RerankConfig,
    ml_Reranker,
    ml_RerankerCreateInput,
    ml_RerankerRerankInput,
    ml_RerankerRerankOutput,
)
from .geniex_sdk.types import ModelConfig, ProfileData

logger = logging.getLogger(__name__)


class Reranker(ModelLoaderMixin):
    """Reranker model wrapper."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ):
        """
        Create and initialize a Reranker instance.

        Args:
            model_path: Path to the model file.
            tokenizer_path: Path to the tokenizer file. If None, uses model_path.
            model_name: Name of the model. If None, uses model_path.
            config: Model configuration. If None, uses default config.
            plugin_id: Plugin to use for the model. If None, uses default.
            device_id: Device to use for the model. If None, uses default device.
        """
        if tokenizer_path is None:
            tokenizer_path = model_path

        if model_name is None:
            model_name = model_path

        if not model_name:
            model_name = model_path

        model_name_bytes = model_name.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        tokenizer_path_bytes = tokenizer_path.encode('utf-8') if tokenizer_path else b''
        c_config = config.to_c_struct() if config else CModelConfig()
        plugin_id_str = plugin_id if plugin_id else ''
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)
        tokenizer_path_buf = ctypes.create_string_buffer(tokenizer_path_bytes) if tokenizer_path_bytes else None

        c_input = ml_RerankerCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            tokenizer_path=ctypes.cast(tokenizer_path_buf, ctypes.c_char_p) if tokenizer_path_buf else None,
            config=c_config,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
        )

        self._string_refs = [
            model_name_buf,
            model_path_buf,
            tokenizer_path_buf,
            plugin_id_buf,
            device_id_buf,
        ]

        handle_ptr = ctypes.POINTER(ml_Reranker)()
        error_code = _lib.ml_reranker_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create Reranker instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy Reranker instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_reranker_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during Reranker cleanup: {e}')
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'reranker'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """Extract Reranker-specific parameters from manifest."""
        tokenizer_path = None
        if manifest.tokenizer_file.name:
            tokenizer_path_obj = store.modelfile_path(repo_id, manifest.tokenizer_file.name)
            if tokenizer_path_obj.exists():
                tokenizer_path = str(tokenizer_path_obj)

        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

        plugin_id = manifest.plugin_id.strip() if manifest.plugin_id else None
        if not plugin_id:
            plugin_id = kwargs.pop('plugin_id', None)
        else:
            # Remove plugin_id from kwargs if manifest has it (to avoid passing it twice)
            kwargs.pop('plugin_id', None)
        if plugin_id:
            plugin_id = plugin_id.strip() if isinstance(plugin_id, str) else plugin_id
            if not plugin_id:
                plugin_id = None

        if not plugin_id:
            plugin_id = cls._get_default_plugin_id()

        device_id = manifest.device_id.strip() if manifest.device_id else None
        if not device_id:
            device_id = kwargs.pop('device_id', None)
        if device_id:
            device_id = device_id.strip() if isinstance(device_id, str) else device_id
            if not device_id:
                device_id = None

        return {
            'model_path': str(model_path),
            'tokenizer_path': tokenizer_path,
            'model_name': model_name,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'Reranker':
        """Create Reranker instance with given parameters."""
        return cls(**params)

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        normalize_method: Optional[str] = 'softmax',
    ) -> 'RerankResult':
        """
        Rerank documents based on query relevance.

        Args:
            query: Query text.
            documents: List of document texts to rerank.
            batch_size: Batch size for processing.
            normalize: Whether to normalize scores.
            normalize_method: Normalization method.

        Returns:
            RerankResult containing scores and metadata.

        Raises:
            Geniex: If reranking fails.
        """
        batch_size = len(documents) if batch_size is None else batch_size
        # C interface implementation
        rerank_config = ml_RerankConfig(
            batch_size=ctypes.c_int32(batch_size),
            normalize=ctypes.c_bool(normalize),
            normalize_method=normalize_method.encode('utf-8') if normalize_method else None,
        )

        query_bytes = query.encode('utf-8')
        query_buf = ctypes.create_string_buffer(query_bytes)

        documents_count = len(documents)
        documents_array = (ctypes.c_char_p * documents_count)()
        doc_bufs = []
        for i, doc in enumerate(documents):
            doc_bytes = doc.encode('utf-8')
            doc_buf = ctypes.create_string_buffer(doc_bytes)
            doc_bufs.append(doc_buf)
            documents_array[i] = ctypes.cast(doc_buf, ctypes.c_char_p)
        documents_ptr = ctypes.cast(documents_array, ctypes.POINTER(ctypes.c_char_p))
        self._doc_bufs = doc_bufs

        c_input = ml_RerankerRerankInput(
            query=ctypes.cast(query_buf, ctypes.c_char_p),
            documents=documents_ptr,
            documents_count=ctypes.c_int32(documents_count),
            config=ctypes.pointer(rerank_config),
        )

        c_output = ml_RerankerRerankOutput()
        error_code = _lib.ml_reranker_rerank(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def _extract_result(self, c_output: ml_RerankerRerankOutput) -> 'RerankResult':
        """Extract RerankResult from C output structure."""
        scores = []
        if c_output.scores and c_output.score_count > 0:
            scores = [c_output.scores[i] for i in range(c_output.score_count)]

        profile_data = ProfileData.from_c_struct(c_output.profile_data)
        return RerankResult(
            scores=scores,
            profile_data=profile_data,
        )


class RerankResult:
    """Result of reranking operation."""

    def __init__(
        self,
        scores: List[float],
        profile_data: ProfileData,
    ):
        self.scores = scores
        self.profile_data = profile_data

    def __repr__(self) -> str:
        return f'RerankResult(scores={len(self.scores)})'
