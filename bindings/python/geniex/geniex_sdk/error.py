"""
Error codes, log levels, and error-related definitions for nexa-sdk.
This module contains C-level definitions (error codes, log levels, callbacks).
"""

import ctypes
from typing import Optional

from ._lib import _lib

# Error codes from ml.h
ML_SUCCESS = 0
ML_ERROR_COMMON_UNKNOWN = -100000
ML_ERROR_COMMON_INVALID_INPUT = -100001
ML_ERROR_COMMON_MEMORY_ALLOCATION = -100003
ML_ERROR_COMMON_FILE_NOT_FOUND = -100004
ML_ERROR_COMMON_NOT_INITIALIZED = -100007
ML_ERROR_COMMON_NOT_SUPPORTED = -100013
ML_ERROR_COMMON_MODEL_LOAD = -100201
ML_ERROR_COMMON_MODEL_INVALID = -100203
ML_ERROR_COMMON_LICENSE_INVALID = -100601
ML_ERROR_COMMON_LICENSE_EXPIRED = -100602

# LLM ERRORS (200xxx)
ML_ERROR_LLM_TOKENIZATION_FAILED = -200001
ML_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH = -200004
ML_ERROR_LLM_GENERATION_FAILED = -200101
ML_ERROR_LLM_GENERATION_PROMPT_TOO_LONG = -200103

# VLM ERRORS (201xxx)
ML_ERROR_VLM_IMAGE_LOAD = -201001
ML_ERROR_VLM_IMAGE_FORMAT = -201002
ML_ERROR_VLM_AUDIO_LOAD = -201101
ML_ERROR_VLM_AUDIO_FORMAT = -201102
ML_ERROR_VLM_GENERATION_FAILED = -201201

# Embedding ERRORS (202xxx)
ML_ERROR_EMBEDDING_GENERATION = -202301
ML_ERROR_EMBEDDING_DIMENSION = -202302

# Reranking ERRORS (203xxx)
ML_ERROR_RERANK_FAILED = -203401
ML_ERROR_RERANK_INPUT = -203402

# Image Generation ERRORS (204xxx)
ML_ERROR_IMAGEGEN_GENERATION = -204501
ML_ERROR_IMAGEGEN_PROMPT = -204502
ML_ERROR_IMAGEGEN_DIMENSION = -204503

# ASR ERRORS (205xxx)
ML_ERROR_ASR_TRANSCRIPTION = -205001
ML_ERROR_ASR_AUDIO_FORMAT = -205002
ML_ERROR_ASR_LANGUAGE = -205003
ML_ERROR_ASR_STREAM_NOT_STARTED = -205010
ML_ERROR_ASR_STREAM_ALREADY_ACTIVE = -205011
ML_ERROR_ASR_STREAM_INVALID_AUDIO = -205012
ML_ERROR_ASR_STREAM_BUFFER_FULL = -205013
ML_ERROR_ASR_STREAM_CALLBACK_ERROR = -205014

# TTS ERRORS (206xxx)
ML_ERROR_TTS_SYNTHESIS = -206001
ML_ERROR_TTS_VOICE = -206002
ML_ERROR_TTS_AUDIO_FORMAT = -206003

# CV ERRORS (207xxx)
ML_ERROR_CV_OCR_DETECTION = -207001
ML_ERROR_CV_OCR_RECOGNITION = -207002
ML_ERROR_CV_OCR_FAILED = -207003

# Diarization ERRORS (208xxx)
ML_ERROR_DIARIZE_AUDIO_LOAD = -208001
ML_ERROR_DIARIZE_SEGMENTATION = -208101
ML_ERROR_DIARIZE_EMBEDDING = -208102
ML_ERROR_DIARIZE_CLUSTERING = -208103

# Log levels
ML_LOG_LEVEL_TRACE = 0
ML_LOG_LEVEL_DEBUG = 1
ML_LOG_LEVEL_INFO = 2
ML_LOG_LEVEL_WARN = 3
ML_LOG_LEVEL_ERROR = 4

# Callback types
ml_log_callback = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p)

# Function signatures
_lib.ml_get_error_message.argtypes = [ctypes.c_int32]
_lib.ml_get_error_message.restype = ctypes.c_char_p


# ============================================================================
# Python Exception Classes
# ============================================================================

# ============================================================================
# Error Handling Functions
# ============================================================================


def ml_get_error_message(error_code: int) -> str:
    """
    Get error message string for error code.

    Args:
        error_code: The error code.

    Returns:
        Error message string.
    """
    result = _lib.ml_get_error_message(error_code)
    if result:
        return result.decode('utf-8')
    return f'Unknown error code: {error_code}'


def check_error(error_code: int) -> None:
    """
    Check error code and raise appropriate exception if not success.

    Args:
        error_code: The error code returned from a C function.

    Raises:
        Geniex: Appropriate exception based on error code.
    """
    if error_code == ML_SUCCESS:
        return

    # Get the exception class from the map, or use base Geniex
    exception_class = _ERROR_MAP.get(error_code, Geniex)
    raise exception_class(error_code)


# ============================================================================
# Python Exception Classes
# ============================================================================


class Geniex(Exception):
    """Base exception for all nexa-sdk errors."""

    def __init__(self, error_code: int, message: Optional[str] = None):
        self.error_code = error_code
        if message is None:
            message = ml_get_error_message(error_code)
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return f'[{self.error_code}] {self.message}'


class Geniex(Geniex):
    """Base class for common errors."""

    pass


class Geniex(Geniex):
    """Unknown error."""

    pass


class Geniex(Geniex):
    """Invalid input parameters or handle."""

    pass


class Geniex(Geniex):
    """Memory allocation failed."""

    pass


class Geniex(Geniex):
    """File not found or inaccessible."""

    pass


class Geniex(Geniex):
    """Library not initialized."""

    pass


class Geniex(Geniex):
    """Operation not supported."""

    pass


class Geniex(Geniex):
    """Model loading failed."""

    pass


class Geniex(Geniex):
    """Invalid model format."""

    pass


class Geniex(Geniex):
    """Invalid license."""

    pass


class Geniex(Geniex):
    """License expired."""

    pass


class Geniex(Geniex):
    """Base class for LLM errors."""

    pass


class Geniex(Geniex):
    """Tokenization failed."""

    pass


class Geniex(Geniex):
    """Context length exceeded."""

    pass


class Geniex(Geniex):
    """Text generation failed."""

    pass


class Geniex(Geniex):
    """Input prompt too long."""

    pass


class Geniex(Geniex):
    """Base class for VLM errors."""

    pass


class Geniex(Geniex):
    """Image loading failed."""

    pass


class Geniex(Geniex):
    """Unsupported image format."""

    pass


class Geniex(Geniex):
    """Audio loading failed."""

    pass


class Geniex(Geniex):
    """Unsupported audio format."""

    pass


class Geniex(Geniex):
    """Multimodal generation failed."""

    pass


class Geniex(Geniex):
    """Base class for Embedding errors."""

    pass


class Geniex(Geniex):
    """Embedding generation failed."""

    pass


class Geniex(Geniex):
    """Invalid embedding dimension."""

    pass


class Geniex(Geniex):
    """Base class for Reranking errors."""

    pass


class Geniex(Geniex):
    """Reranking failed."""

    pass


class Geniex(Geniex):
    """Invalid reranking input."""

    pass


class Geniex(Geniex):
    """Base class for Image Generation errors."""

    pass


class Geniex(Geniex):
    """Image generation failed."""

    pass


class Geniex(Geniex):
    """Invalid image prompt."""

    pass


class Geniex(Geniex):
    """Invalid image dimensions."""

    pass


class Geniex(Geniex):
    """Base class for ASR errors."""

    pass


class Geniex(Geniex):
    """ASR transcription failed."""

    pass


class Geniex(Geniex):
    """Unsupported ASR audio format."""

    pass


class Geniex(Geniex):
    """Unsupported ASR language."""

    pass


class Geniex(Geniex):
    """Streaming not started."""

    pass


class Geniex(Geniex):
    """Streaming already active."""

    pass


class Geniex(Geniex):
    """Invalid audio data."""

    pass


class Geniex(Geniex):
    """Audio buffer full."""

    pass


class Geniex(Geniex):
    """Callback execution error."""

    pass


class Geniex(Geniex):
    """Base class for TTS errors."""

    pass


class Geniex(Geniex):
    """TTS synthesis failed."""

    pass


class Geniex(Geniex):
    """TTS voice not found."""

    pass


class Geniex(Geniex):
    """TTS audio format error."""

    pass


class Geniex(Geniex):
    """Base class for CV errors."""

    pass


class Geniex(Geniex):
    """OCR text detection failed."""

    pass


class Geniex(Geniex):
    """OCR text recognition failed."""

    pass


class Geniex(Geniex):
    """OCR failed."""

    pass


class Geniex(Geniex):
    """Base class for Diarization errors."""

    pass


class Geniex(Geniex):
    """Audio loading failed."""

    pass


class Geniex(Geniex):
    """Segmentation model execution failed."""

    pass


class Geniex(Geniex):
    """Embedding extraction failed."""

    pass


class Geniex(Geniex):
    """Speaker clustering failed (PLDA/VBx)."""

    pass


# Error code to exception class mapping
_ERROR_MAP = {
    ML_ERROR_COMMON_UNKNOWN: Geniex,
    ML_ERROR_COMMON_INVALID_INPUT: Geniex,
    ML_ERROR_COMMON_MEMORY_ALLOCATION: Geniex,
    ML_ERROR_COMMON_FILE_NOT_FOUND: Geniex,
    ML_ERROR_COMMON_NOT_INITIALIZED: Geniex,
    ML_ERROR_COMMON_NOT_SUPPORTED: Geniex,
    ML_ERROR_COMMON_MODEL_LOAD: Geniex,
    ML_ERROR_COMMON_MODEL_INVALID: Geniex,
    ML_ERROR_COMMON_LICENSE_INVALID: Geniex,
    ML_ERROR_COMMON_LICENSE_EXPIRED: Geniex,
    ML_ERROR_LLM_TOKENIZATION_FAILED: Geniex,
    ML_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH: Geniex,
    ML_ERROR_LLM_GENERATION_FAILED: Geniex,
    ML_ERROR_LLM_GENERATION_PROMPT_TOO_LONG: Geniex,
    ML_ERROR_VLM_IMAGE_LOAD: Geniex,
    ML_ERROR_VLM_IMAGE_FORMAT: Geniex,
    ML_ERROR_VLM_AUDIO_LOAD: Geniex,
    ML_ERROR_VLM_AUDIO_FORMAT: Geniex,
    ML_ERROR_VLM_GENERATION_FAILED: Geniex,
    ML_ERROR_EMBEDDING_GENERATION: Geniex,
    ML_ERROR_EMBEDDING_DIMENSION: Geniex,
    ML_ERROR_RERANK_FAILED: Geniex,
    ML_ERROR_RERANK_INPUT: Geniex,
    ML_ERROR_IMAGEGEN_GENERATION: Geniex,
    ML_ERROR_IMAGEGEN_PROMPT: Geniex,
    ML_ERROR_IMAGEGEN_DIMENSION: Geniex,
    ML_ERROR_ASR_TRANSCRIPTION: Geniex,
    ML_ERROR_ASR_AUDIO_FORMAT: Geniex,
    ML_ERROR_ASR_LANGUAGE: Geniex,
    ML_ERROR_ASR_STREAM_NOT_STARTED: Geniex,
    ML_ERROR_ASR_STREAM_ALREADY_ACTIVE: Geniex,
    ML_ERROR_ASR_STREAM_INVALID_AUDIO: Geniex,
    ML_ERROR_ASR_STREAM_BUFFER_FULL: Geniex,
    ML_ERROR_ASR_STREAM_CALLBACK_ERROR: Geniex,
    ML_ERROR_TTS_SYNTHESIS: Geniex,
    ML_ERROR_TTS_VOICE: Geniex,
    ML_ERROR_TTS_AUDIO_FORMAT: Geniex,
    ML_ERROR_CV_OCR_DETECTION: Geniex,
    ML_ERROR_CV_OCR_RECOGNITION: Geniex,
    ML_ERROR_CV_OCR_FAILED: Geniex,
    ML_ERROR_DIARIZE_AUDIO_LOAD: Geniex,
    ML_ERROR_DIARIZE_SEGMENTATION: Geniex,
    ML_ERROR_DIARIZE_EMBEDDING: Geniex,
    ML_ERROR_DIARIZE_CLUSTERING: Geniex,
}
