"""
Embedding functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig, ml_ProfileData

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_EmbeddingConfig(ctypes.Structure):
    """Embedding generation configuration."""

    _fields_ = [
        ('batch_size', ctypes.c_int32),
        ('normalize', ctypes.c_bool),
        ('normalize_method', ctypes.c_char_p),
    ]


class ml_Embedder(ctypes.Structure):
    """Opaque embedder handle."""

    pass


class ml_EmbedderCreateInput(ctypes.Structure):
    """Input structure for creating an embedder."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('tokenizer_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
    ]


class ml_EmbedderEmbedInput(ctypes.Structure):
    """Input structure for embedding generation."""

    _fields_ = [
        ('texts', ctypes.POINTER(ctypes.c_char_p)),
        ('text_count', ctypes.c_int32),
        ('config', ctypes.POINTER(ml_EmbeddingConfig)),
        ('input_ids_2d', ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
        ('input_ids_row_lengths', ctypes.POINTER(ctypes.c_int32)),
        ('input_ids_row_count', ctypes.c_int32),
        ('task_type', ctypes.c_char_p),
        ('image_paths', ctypes.POINTER(ctypes.c_char_p)),
        ('image_count', ctypes.c_int32),
    ]


class ml_EmbedderEmbedOutput(ctypes.Structure):
    """Output structure for embedding generation."""

    _fields_ = [
        ('embeddings', ctypes.POINTER(ctypes.c_float)),
        ('embedding_count', ctypes.c_int32),
        ('profile_data', ml_ProfileData),
    ]


class ml_EmbedderDimOutput(ctypes.Structure):
    """Output structure for getting embedding dimension."""

    _fields_ = [
        ('dimension', ctypes.c_int32),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_embedder_create.argtypes = [
    ctypes.POINTER(ml_EmbedderCreateInput),
    ctypes.POINTER(ctypes.POINTER(ml_Embedder)),
]
_lib.ml_embedder_create.restype = ctypes.c_int32

_lib.ml_embedder_destroy.argtypes = [ctypes.POINTER(ml_Embedder)]
_lib.ml_embedder_destroy.restype = ctypes.c_int32

_lib.ml_embedder_embed.argtypes = [
    ctypes.POINTER(ml_Embedder),
    ctypes.POINTER(ml_EmbedderEmbedInput),
    ctypes.POINTER(ml_EmbedderEmbedOutput),
]
_lib.ml_embedder_embed.restype = ctypes.c_int32

_lib.ml_embedder_embedding_dim.argtypes = [ctypes.POINTER(ml_Embedder), ctypes.POINTER(ml_EmbedderDimOutput)]
_lib.ml_embedder_embedding_dim.restype = ctypes.c_int32
