"""
Rerank functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig, ml_ProfileData

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_RerankConfig(ctypes.Structure):
    """Reranking configuration."""

    _fields_ = [
        ('batch_size', ctypes.c_int32),
        ('normalize', ctypes.c_bool),
        ('normalize_method', ctypes.c_char_p),
    ]


class ml_Reranker(ctypes.Structure):
    """Opaque reranker handle."""

    pass


class ml_RerankerCreateInput(ctypes.Structure):
    """Input structure for creating a reranker."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('tokenizer_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
    ]


class ml_RerankerRerankInput(ctypes.Structure):
    """Input structure for reranking operation."""

    _fields_ = [
        ('query', ctypes.c_char_p),
        ('documents', ctypes.POINTER(ctypes.c_char_p)),
        ('documents_count', ctypes.c_int32),
        ('config', ctypes.POINTER(ml_RerankConfig)),
    ]


class ml_RerankerRerankOutput(ctypes.Structure):
    """Output structure for reranking operation."""

    _fields_ = [
        ('scores', ctypes.POINTER(ctypes.c_float)),
        ('score_count', ctypes.c_int32),
        ('profile_data', ml_ProfileData),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_reranker_create.argtypes = [
    ctypes.POINTER(ml_RerankerCreateInput),
    ctypes.POINTER(ctypes.POINTER(ml_Reranker)),
]
_lib.ml_reranker_create.restype = ctypes.c_int32

_lib.ml_reranker_destroy.argtypes = [ctypes.POINTER(ml_Reranker)]
_lib.ml_reranker_destroy.restype = ctypes.c_int32

_lib.ml_reranker_rerank.argtypes = [
    ctypes.POINTER(ml_Reranker),
    ctypes.POINTER(ml_RerankerRerankInput),
    ctypes.POINTER(ml_RerankerRerankOutput),
]
_lib.ml_reranker_rerank.restype = ctypes.c_int32
