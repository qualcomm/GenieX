"""
VLM functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_GenerationConfig, ml_ModelConfig, ml_ProfileData, ml_token_callback

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_VlmContent(ctypes.Structure):
    """VLM content structure."""

    _fields_ = [
        ('type', ctypes.c_char_p),
        ('text', ctypes.c_char_p),
    ]


class ml_VlmChatMessage(ctypes.Structure):
    """VLM chat message structure."""

    _fields_ = [
        ('role', ctypes.c_char_p),
        ('contents', ctypes.POINTER(ml_VlmContent)),
        ('content_count', ctypes.c_int64),
    ]


class ml_VLM(ctypes.Structure):
    """Opaque VLM handle."""

    pass


class ml_VlmCreateInput(ctypes.Structure):
    """Input structure for creating a VLM instance."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('mmproj_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
        ('tokenizer_path', ctypes.c_char_p),
        ('license_id', ctypes.c_char_p),
        ('license_key', ctypes.c_char_p),
    ]


class ml_VlmApplyChatTemplateInput(ctypes.Structure):
    """Input structure for applying VLM chat template."""

    _fields_ = [
        ('messages', ctypes.POINTER(ml_VlmChatMessage)),
        ('message_count', ctypes.c_int32),
        ('tools', ctypes.c_char_p),
        ('enable_thinking', ctypes.c_bool),
        ('grounding', ctypes.c_bool),
    ]


class ml_VlmApplyChatTemplateOutput(ctypes.Structure):
    """Output structure for applying VLM chat template."""

    _fields_ = [
        ('formatted_text', ctypes.c_void_p),
    ]


class ml_VlmGenerateInput(ctypes.Structure):
    """Input structure for VLM streaming text generation."""

    _fields_ = [
        ('prompt_utf8', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_GenerationConfig)),
        ('on_token', ml_token_callback),
        ('user_data', ctypes.c_void_p),
    ]


class ml_VlmGenerateOutput(ctypes.Structure):
    """Output structure for VLM streaming text generation."""

    _fields_ = [
        ('full_text', ctypes.c_void_p),
        ('profile_data', ml_ProfileData),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_vlm_create.argtypes = [ctypes.POINTER(ml_VlmCreateInput), ctypes.POINTER(ctypes.POINTER(ml_VLM))]
_lib.ml_vlm_create.restype = ctypes.c_int32

_lib.ml_vlm_destroy.argtypes = [ctypes.POINTER(ml_VLM)]
_lib.ml_vlm_destroy.restype = ctypes.c_int32

_lib.ml_vlm_reset.argtypes = [ctypes.POINTER(ml_VLM)]
_lib.ml_vlm_reset.restype = ctypes.c_int32

_lib.ml_vlm_apply_chat_template.argtypes = [
    ctypes.POINTER(ml_VLM),
    ctypes.POINTER(ml_VlmApplyChatTemplateInput),
    ctypes.POINTER(ml_VlmApplyChatTemplateOutput),
]
_lib.ml_vlm_apply_chat_template.restype = ctypes.c_int32

_lib.ml_vlm_generate.argtypes = [
    ctypes.POINTER(ml_VLM),
    ctypes.POINTER(ml_VlmGenerateInput),
    ctypes.POINTER(ml_VlmGenerateOutput),
]
_lib.ml_vlm_generate.restype = ctypes.c_int32
