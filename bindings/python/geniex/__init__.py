"""
Geniex - Python SDK for Nexa AI ML library.

This package provides a Pythonic interface to the nexa-sdk machine learning library,
including support for LLMs, embeddings, and other ML operations.
"""

import atexit

from ._version import __version__
from .asr import ASR, TranscribeResult
from .core import _init_logging, get_device_list, get_plugin_list, setup_logging
from .cv import CV, BoundingBox, CVResult, CVResultItem
from .diarize import Diarize, DiarizeResult, SpeechSegment
from .embedding import Embedder, EmbedResult
from .image_gen import ImageGen, ImageGenResult
from .internal.types import DownloadProgressInfo, ModelInfo
from .llm import LLM, GenerateResult
from .models import download_model, list_models, remove_model
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import (
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    Geniex,
    check_error,
    ml_get_error_message,
)
from .geniex_sdk.types import (
    GenerationConfig,
    KvCacheLoadInput,
    KvCacheSaveInput,
    LlmChatMessage,
    ModelConfig,
    ProfileData,
    SamplerConfig,
    VlmChatMessage,
    VlmContent,
)
from .rerank import Reranker, RerankResult
from .tts import TTS, SynthesizeResult
from .vlm import VLM


def version() -> str:
    """Get Python package version."""
    return __version__


def geniex_version() -> str:
    """Get Nexa SDK (C library) version."""
    return _lib.ml_version().decode('utf-8')


_init_logging()

check_error(_lib.ml_init())
atexit.register(lambda: check_error(_lib.ml_deinit()))

__all__ = [
    # Core functions
    'version',
    'geniex_version',
    'get_plugin_list',
    'get_device_list',
    # Logging
    'setup_logging',
    # Model management
    'download_model',
    'list_models',
    'remove_model',
    'ModelInfo',
    'DownloadProgressInfo',
    'FileProgressInfo',
    # LLM
    'LLM',
    'GenerateResult',
    # VLM
    'VLM',
    # ASR
    'ASR',
    'TranscribeResult',
    # TTS
    'TTS',
    'SynthesizeResult',
    # Embedding
    'Embedder',
    'EmbedResult',
    # Rerank
    'Reranker',
    'RerankResult',
    # Diarize
    'Diarize',
    'DiarizeResult',
    'SpeechSegment',
    # CV
    'CV',
    'CVResult',
    'CVResultItem',
    'BoundingBox',
    # ImageGen
    'ImageGen',
    'ImageGenResult',
    # Types
    'SamplerConfig',
    'GenerationConfig',
    'ModelConfig',
    'LlmChatMessage',
    'VlmChatMessage',
    'VlmContent',
    'ProfileData',
    'KvCacheSaveInput',
    'KvCacheLoadInput',
    # Errors
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'Geniex',
    'check_error',
    'ml_get_error_message',
]
