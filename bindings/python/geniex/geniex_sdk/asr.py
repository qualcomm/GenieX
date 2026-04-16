"""
ASR functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig, ml_ProfileData

# ============================================================================
# C Structure Definitions
# ============================================================================

ml_asr_transcription_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_void_p)


class ml_ASRConfig(ctypes.Structure):
    """ASR processing configuration."""

    _fields_ = [
        ('timestamps', ctypes.c_char_p),
        ('beam_size', ctypes.c_int32),
        ('stream', ctypes.c_bool),
    ]


class ml_ASRResult(ctypes.Structure):
    """ASR transcription result."""

    _fields_ = [
        ('transcript', ctypes.c_char_p),
        ('confidence_scores', ctypes.POINTER(ctypes.c_float)),
        ('confidence_count', ctypes.c_int32),
        ('timestamps', ctypes.POINTER(ctypes.c_float)),
        ('timestamp_count', ctypes.c_int32),
    ]


class ml_ASR(ctypes.Structure):
    """Opaque ASR handle."""

    pass


class ml_AsrCreateInput(ctypes.Structure):
    """Input structure for creating an ASR instance."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('tokenizer_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('language', ctypes.c_char_p),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
        ('license_id', ctypes.c_char_p),
        ('license_key', ctypes.c_char_p),
    ]


class ml_AsrTranscribeInput(ctypes.Structure):
    """Input structure for ASR transcription."""

    _fields_ = [
        ('audio_path', ctypes.c_char_p),
        ('language', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_ASRConfig)),
    ]


class ml_AsrTranscribeOutput(ctypes.Structure):
    """Output structure for ASR transcription."""

    _fields_ = [
        ('result', ml_ASRResult),
        ('profile_data', ml_ProfileData),
    ]


class ml_AsrListSupportedLanguagesInput(ctypes.Structure):
    """Input structure for getting supported languages."""

    _fields_ = [
        ('reserved', ctypes.c_void_p),
    ]


class ml_AsrListSupportedLanguagesOutput(ctypes.Structure):
    """Output structure for getting supported languages."""

    _fields_ = [
        ('language_codes', ctypes.POINTER(ctypes.c_char_p)),
        ('language_count', ctypes.c_int32),
    ]


class ml_ASRStreamConfig(ctypes.Structure):
    """ASR streaming configuration."""

    _fields_ = [
        ('chunk_duration', ctypes.c_float),
        ('overlap_duration', ctypes.c_float),
        ('sample_rate', ctypes.c_int32),
        ('max_queue_size', ctypes.c_int32),
        ('buffer_size', ctypes.c_int32),
        ('timestamps', ctypes.c_char_p),
        ('beam_size', ctypes.c_int32),
    ]


class ml_AsrStreamBeginInput(ctypes.Structure):
    """Input structure for beginning ASR streaming."""

    _fields_ = [
        ('stream_config', ctypes.POINTER(ml_ASRStreamConfig)),
        ('language', ctypes.c_char_p),
        ('on_transcription', ml_asr_transcription_callback),
        ('user_data', ctypes.c_void_p),
    ]


class ml_AsrStreamBeginOutput(ctypes.Structure):
    """Output structure for streaming begin."""

    _fields_ = [
        ('reserved', ctypes.c_void_p),
    ]


class ml_AsrStreamPushAudioInput(ctypes.Structure):
    """Input structure for processing audio data."""

    _fields_ = [
        ('audio_data', ctypes.POINTER(ctypes.c_float)),
        ('length', ctypes.c_int32),
    ]


class ml_AsrStreamStopInput(ctypes.Structure):
    """Input structure for stopping streaming."""

    _fields_ = [
        ('graceful', ctypes.c_bool),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_asr_create.argtypes = [ctypes.POINTER(ml_AsrCreateInput), ctypes.POINTER(ctypes.POINTER(ml_ASR))]
_lib.ml_asr_create.restype = ctypes.c_int32

_lib.ml_asr_destroy.argtypes = [ctypes.POINTER(ml_ASR)]
_lib.ml_asr_destroy.restype = ctypes.c_int32

_lib.ml_asr_transcribe.argtypes = [
    ctypes.POINTER(ml_ASR),
    ctypes.POINTER(ml_AsrTranscribeInput),
    ctypes.POINTER(ml_AsrTranscribeOutput),
]
_lib.ml_asr_transcribe.restype = ctypes.c_int32

_lib.ml_asr_list_supported_languages.argtypes = [
    ctypes.POINTER(ml_ASR),
    ctypes.POINTER(ml_AsrListSupportedLanguagesInput),
    ctypes.POINTER(ml_AsrListSupportedLanguagesOutput),
]
_lib.ml_asr_list_supported_languages.restype = ctypes.c_int32

_lib.ml_asr_stream_begin.argtypes = [
    ctypes.POINTER(ml_ASR),
    ctypes.POINTER(ml_AsrStreamBeginInput),
    ctypes.POINTER(ml_AsrStreamBeginOutput),
]
_lib.ml_asr_stream_begin.restype = ctypes.c_int32

_lib.ml_asr_stream_push_audio.argtypes = [ctypes.POINTER(ml_ASR), ctypes.POINTER(ml_AsrStreamPushAudioInput)]
_lib.ml_asr_stream_push_audio.restype = ctypes.c_int32

_lib.ml_asr_stream_stop.argtypes = [ctypes.POINTER(ml_ASR), ctypes.POINTER(ml_AsrStreamStopInput)]
_lib.ml_asr_stream_stop.restype = ctypes.c_int32
