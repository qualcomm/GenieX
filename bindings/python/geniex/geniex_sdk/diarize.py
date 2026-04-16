"""
Diarize functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig, ml_ProfileData

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_DiarizeConfig(ctypes.Structure):
    """Diarization processing configuration."""

    _fields_ = [
        ('min_speakers', ctypes.c_int32),
        ('max_speakers', ctypes.c_int32),
    ]


class ml_DiarizeSpeechSegment(ctypes.Structure):
    """Speech segment structure."""

    _fields_ = [
        ('start_time', ctypes.c_float),
        ('end_time', ctypes.c_float),
        ('speaker_label', ctypes.c_char_p),
    ]


class ml_Diarize(ctypes.Structure):
    """Opaque diarization handle."""

    pass


class ml_DiarizeCreateInput(ctypes.Structure):
    """Input structure for creating a diarization instance."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
        ('license_id', ctypes.c_char_p),
        ('license_key', ctypes.c_char_p),
    ]


class ml_DiarizeInferInput(ctypes.Structure):
    """Input structure for diarization inference."""

    _fields_ = [
        ('audio_path', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_DiarizeConfig)),
    ]


class ml_DiarizeInferOutput(ctypes.Structure):
    """Output structure for diarization inference."""

    _fields_ = [
        ('segments', ctypes.POINTER(ml_DiarizeSpeechSegment)),
        ('segment_count', ctypes.c_int32),
        ('num_speakers', ctypes.c_int32),
        ('duration', ctypes.c_float),
        ('profile_data', ml_ProfileData),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_diarize_create.argtypes = [
    ctypes.POINTER(ml_DiarizeCreateInput),
    ctypes.POINTER(ctypes.POINTER(ml_Diarize)),
]
_lib.ml_diarize_create.restype = ctypes.c_int32

_lib.ml_diarize_destroy.argtypes = [ctypes.POINTER(ml_Diarize)]
_lib.ml_diarize_destroy.restype = ctypes.c_int32

_lib.ml_diarize_infer.argtypes = [
    ctypes.POINTER(ml_Diarize),
    ctypes.POINTER(ml_DiarizeInferInput),
    ctypes.POINTER(ml_DiarizeInferOutput),
]
_lib.ml_diarize_infer.restype = ctypes.c_int32
