"""
Nexa SDK - Python wrapper for nexa-sdk C library.

This module provides low-level ctypes bindings to the nexa-sdk C library.
All Python implementations are in geniex and internal packages.
"""

# Export library loader
from ._lib import _lib

# Export ASR structures and types
from .asr import (
    ml_ASR,
    ml_asr_transcription_callback,
    ml_ASRConfig,
    ml_AsrCreateInput,
    ml_AsrListSupportedLanguagesInput,
    ml_AsrListSupportedLanguagesOutput,
    ml_ASRResult,
    ml_AsrStreamBeginInput,
    ml_AsrStreamBeginOutput,
    ml_ASRStreamConfig,
    ml_AsrStreamPushAudioInput,
    ml_AsrStreamStopInput,
    ml_AsrTranscribeInput,
    ml_AsrTranscribeOutput,
)

# Export CV structures and types
from .cv import (
    ml_BoundingBox,
    ml_CV,
    ml_CVCapabilities,
    ml_CVCreateInput,
    ml_CVInferInput,
    ml_CVInferOutput,
    ml_CVModelConfig,
    ml_CVResult,
)

# Export Diarize structures and types
from .diarize import (
    ml_Diarize,
    ml_DiarizeConfig,
    ml_DiarizeCreateInput,
    ml_DiarizeInferInput,
    ml_DiarizeInferOutput,
    ml_DiarizeSpeechSegment,
)

# Export Embedding structures and types
from .embedding import (
    ml_Embedder,
    ml_EmbedderCreateInput,
    ml_EmbedderDimOutput,
    ml_EmbedderEmbedInput,
    ml_EmbedderEmbedOutput,
    ml_EmbeddingConfig,
)

# Export error codes and definitions
# Export error classes (from error.py)
from .error import (
    _ERROR_MAP,
    ML_ERROR_COMMON_FILE_NOT_FOUND,
    ML_ERROR_COMMON_INVALID_INPUT,
    ML_ERROR_COMMON_LICENSE_EXPIRED,
    ML_ERROR_COMMON_LICENSE_INVALID,
    ML_ERROR_COMMON_MEMORY_ALLOCATION,
    ML_ERROR_COMMON_MODEL_INVALID,
    ML_ERROR_COMMON_MODEL_LOAD,
    ML_ERROR_COMMON_NOT_INITIALIZED,
    ML_ERROR_COMMON_NOT_SUPPORTED,
    ML_ERROR_COMMON_UNKNOWN,
    ML_ERROR_LLM_GENERATION_FAILED,
    ML_ERROR_LLM_GENERATION_PROMPT_TOO_LONG,
    ML_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH,
    ML_ERROR_LLM_TOKENIZATION_FAILED,
    ML_LOG_LEVEL_DEBUG,
    ML_LOG_LEVEL_ERROR,
    ML_LOG_LEVEL_INFO,
    ML_LOG_LEVEL_TRACE,
    ML_LOG_LEVEL_WARN,
    ML_SUCCESS,
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
    ml_log_callback,
)

# Export ImageGen structures and types
from .image_gen import (
    ml_ImageGen,
    ml_ImageGenCreateInput,
    ml_ImageGenerationConfig,
    ml_ImageGenImg2ImgInput,
    ml_ImageGenOutput,
    ml_ImageGenTxt2ImgInput,
    ml_ImageSamplerConfig,
    ml_SchedulerConfig,
)

# Export LLM structures and types
from .llm import (
    ml_GenerationConfig,
    ml_KvCacheLoadInput,
    ml_KvCacheLoadOutput,
    ml_KvCacheSaveInput,
    ml_KvCacheSaveOutput,
    ml_LLM,
    ml_LlmApplyChatTemplateInput,
    ml_LlmApplyChatTemplateOutput,
    ml_LlmChatMessage,
    ml_LlmCreateInput,
    ml_LlmGenerateInput,
    ml_LlmGenerateOutput,
    ml_ModelConfig,
    ml_ProfileData,
    ml_SamplerConfig,
    ml_token_callback,
)

# Export ML core structures and types
from .ml import (
    ml_create_plugin_func,
    ml_GetDeviceListInput,
    ml_GetDeviceListOutput,
    ml_GetPluginListOutput,
    ml_plugin_id_func,
)

# Export Rerank structures and types
from .rerank import (
    ml_RerankConfig,
    ml_Reranker,
    ml_RerankerCreateInput,
    ml_RerankerRerankInput,
    ml_RerankerRerankOutput,
)

# Export TTS structures and types
from .tts import (
    ml_TTS,
    ml_TTSConfig,
    ml_TtsCreateInput,
    ml_TtsListAvailableVoicesInput,
    ml_TtsListAvailableVoicesOutput,
    ml_TTSResult,
    ml_TTSSamplerConfig,
    ml_TtsSynthesizeInput,
    ml_TtsSynthesizeOutput,
)

# Export types
from .types import (
    GenerationConfig,
    KvCacheLoadInput,
    KvCacheSaveInput,
    LlmChatMessage,
    ModelConfig,
    ProfileData,
    SamplerConfig,
)

# Export VLM structures and types
from .vlm import (
    ml_VLM,
    ml_VlmApplyChatTemplateInput,
    ml_VlmApplyChatTemplateOutput,
    ml_VlmChatMessage,
    ml_VlmContent,
    ml_VlmCreateInput,
    ml_VlmGenerateInput,
    ml_VlmGenerateOutput,
)

__version__ = '0.1.0'

__all__ = [
    # Library
    '_lib',
    # Error codes
    'ML_SUCCESS',
    'ML_ERROR_COMMON_UNKNOWN',
    'ML_ERROR_COMMON_INVALID_INPUT',
    'ML_ERROR_COMMON_MEMORY_ALLOCATION',
    'ML_ERROR_COMMON_FILE_NOT_FOUND',
    'ML_ERROR_COMMON_NOT_INITIALIZED',
    'ML_ERROR_COMMON_NOT_SUPPORTED',
    'ML_ERROR_COMMON_MODEL_LOAD',
    'ML_ERROR_COMMON_MODEL_INVALID',
    'ML_ERROR_COMMON_LICENSE_INVALID',
    'ML_ERROR_COMMON_LICENSE_EXPIRED',
    'ML_ERROR_LLM_TOKENIZATION_FAILED',
    'ML_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH',
    'ML_ERROR_LLM_GENERATION_FAILED',
    'ML_ERROR_LLM_GENERATION_PROMPT_TOO_LONG',
    'ML_LOG_LEVEL_TRACE',
    'ML_LOG_LEVEL_DEBUG',
    'ML_LOG_LEVEL_INFO',
    'ML_LOG_LEVEL_WARN',
    'ML_LOG_LEVEL_ERROR',
    'ml_log_callback',
    # ML core types
    'ml_plugin_id_func',
    'ml_create_plugin_func',
    'ml_GetPluginListOutput',
    'ml_GetDeviceListInput',
    'ml_GetDeviceListOutput',
    # LLM types
    'ml_SamplerConfig',
    'ml_GenerationConfig',
    'ml_ModelConfig',
    'ml_LLM',
    'ml_LlmCreateInput',
    'ml_LlmChatMessage',
    'ml_LlmApplyChatTemplateInput',
    'ml_LlmApplyChatTemplateOutput',
    'ml_ProfileData',
    'ml_LlmGenerateInput',
    'ml_LlmGenerateOutput',
    'ml_KvCacheSaveInput',
    'ml_KvCacheSaveOutput',
    'ml_KvCacheLoadInput',
    'ml_KvCacheLoadOutput',
    'ml_token_callback',
    # VLM types
    'ml_VLM',
    'ml_VlmContent',
    'ml_VlmChatMessage',
    'ml_VlmCreateInput',
    'ml_VlmApplyChatTemplateInput',
    'ml_VlmApplyChatTemplateOutput',
    'ml_VlmGenerateInput',
    'ml_VlmGenerateOutput',
    # ASR types
    'ml_ASR',
    'ml_ASRConfig',
    'ml_ASRResult',
    'ml_ASRStreamConfig',
    'ml_AsrCreateInput',
    'ml_AsrTranscribeInput',
    'ml_AsrTranscribeOutput',
    'ml_AsrListSupportedLanguagesInput',
    'ml_AsrListSupportedLanguagesOutput',
    'ml_AsrStreamBeginInput',
    'ml_AsrStreamBeginOutput',
    'ml_AsrStreamPushAudioInput',
    'ml_AsrStreamStopInput',
    'ml_asr_transcription_callback',
    # Embedding types
    'ml_Embedder',
    'ml_EmbeddingConfig',
    'ml_EmbedderCreateInput',
    'ml_EmbedderEmbedInput',
    'ml_EmbedderEmbedOutput',
    'ml_EmbedderDimOutput',
    # TTS types
    'ml_TTS',
    'ml_TTSConfig',
    'ml_TTSSamplerConfig',
    'ml_TTSResult',
    'ml_TtsCreateInput',
    'ml_TtsSynthesizeInput',
    'ml_TtsSynthesizeOutput',
    'ml_TtsListAvailableVoicesInput',
    'ml_TtsListAvailableVoicesOutput',
    # Diarize types
    'ml_Diarize',
    'ml_DiarizeConfig',
    'ml_DiarizeSpeechSegment',
    'ml_DiarizeCreateInput',
    'ml_DiarizeInferInput',
    'ml_DiarizeInferOutput',
    # Rerank types
    'ml_Reranker',
    'ml_RerankConfig',
    'ml_RerankerCreateInput',
    'ml_RerankerRerankInput',
    'ml_RerankerRerankOutput',
    # CV types
    'ml_CV',
    'ml_BoundingBox',
    'ml_CVResult',
    'ml_CVCapabilities',
    'ml_CVModelConfig',
    'ml_CVCreateInput',
    'ml_CVInferInput',
    'ml_CVInferOutput',
    # ImageGen types
    'ml_ImageGen',
    'ml_ImageSamplerConfig',
    'ml_SchedulerConfig',
    'ml_ImageGenerationConfig',
    'ml_ImageGenCreateInput',
    'ml_ImageGenTxt2ImgInput',
    'ml_ImageGenImg2ImgInput',
    'ml_ImageGenOutput',
    # Error classes
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
    # Error handling functions
    'check_error',
    'ml_get_error_message',
    # Types
    'SamplerConfig',
    'GenerationConfig',
    'ModelConfig',
    'LlmChatMessage',
    'ProfileData',
    'KvCacheSaveInput',
    'KvCacheLoadInput',
]
