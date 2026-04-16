"""
TTS functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig, ml_ProfileData

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_TTSConfig(ctypes.Structure):
    """TTS synthesis configuration."""

    _fields_ = [
        ('voice', ctypes.c_char_p),
        ('speed', ctypes.c_float),
        ('seed', ctypes.c_int32),
        ('sample_rate', ctypes.c_int32),
    ]


class ml_TTSSamplerConfig(ctypes.Structure):
    """TTS sampling parameters."""

    _fields_ = [
        ('temperature', ctypes.c_float),
        ('noise_scale', ctypes.c_float),
        ('length_scale', ctypes.c_float),
    ]


class ml_TTSResult(ctypes.Structure):
    """TTS synthesis result."""

    _fields_ = [
        ('audio_path', ctypes.c_char_p),
        ('duration_seconds', ctypes.c_float),
        ('sample_rate', ctypes.c_int32),
        ('channels', ctypes.c_int32),
        ('num_samples', ctypes.c_int32),
    ]


class ml_TTS(ctypes.Structure):
    """Opaque TTS handle."""

    pass


class ml_TtsCreateInput(ctypes.Structure):
    """Input structure for creating a TTS instance."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('vocoder_path', ctypes.c_char_p),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
    ]


class ml_TtsSynthesizeInput(ctypes.Structure):
    """Input structure for TTS synthesis."""

    _fields_ = [
        ('text_utf8', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_TTSConfig)),
        ('output_path', ctypes.c_char_p),
    ]


class ml_TtsSynthesizeOutput(ctypes.Structure):
    """Output structure for TTS synthesis."""

    _fields_ = [
        ('result', ml_TTSResult),
        ('profile_data', ml_ProfileData),
    ]


class ml_TtsListAvailableVoicesInput(ctypes.Structure):
    """Input structure for getting available voices."""

    _fields_ = [
        ('reserved', ctypes.c_void_p),
    ]


class ml_TtsListAvailableVoicesOutput(ctypes.Structure):
    """Output structure for getting available voices."""

    _fields_ = [
        ('voice_ids', ctypes.POINTER(ctypes.c_char_p)),
        ('voice_count', ctypes.c_int32),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_tts_create.argtypes = [ctypes.POINTER(ml_TtsCreateInput), ctypes.POINTER(ctypes.POINTER(ml_TTS))]
_lib.ml_tts_create.restype = ctypes.c_int32

_lib.ml_tts_destroy.argtypes = [ctypes.POINTER(ml_TTS)]
_lib.ml_tts_destroy.restype = ctypes.c_int32

_lib.ml_tts_synthesize.argtypes = [
    ctypes.POINTER(ml_TTS),
    ctypes.POINTER(ml_TtsSynthesizeInput),
    ctypes.POINTER(ml_TtsSynthesizeOutput),
]
_lib.ml_tts_synthesize.restype = ctypes.c_int32

_lib.ml_tts_list_available_voices.argtypes = [
    ctypes.POINTER(ml_TTS),
    ctypes.POINTER(ml_TtsListAvailableVoicesInput),
    ctypes.POINTER(ml_TtsListAvailableVoicesOutput),
]
_lib.ml_tts_list_available_voices.restype = ctypes.c_int32
