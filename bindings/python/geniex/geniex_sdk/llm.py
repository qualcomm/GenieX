"""
LLM functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback

# Callback types
ml_token_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_char_p, ctypes.c_void_p)


# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_SamplerConfig(ctypes.Structure):
    """Text generation sampling parameters."""

    _fields_ = [
        ('temperature', ctypes.c_float),
        ('top_p', ctypes.c_float),
        ('top_k', ctypes.c_int32),
        ('min_p', ctypes.c_float),
        ('repetition_penalty', ctypes.c_float),
        ('presence_penalty', ctypes.c_float),
        ('frequency_penalty', ctypes.c_float),
        ('seed', ctypes.c_int32),
        ('grammar_path', ctypes.c_char_p),
        ('grammar_string', ctypes.c_char_p),
        ('enable_json', ctypes.c_bool),
    ]


class ml_GenerationConfig(ctypes.Structure):
    """LLM / VLM generation configuration."""

    _fields_ = [
        ('max_tokens', ctypes.c_int32),
        ('stop', ctypes.POINTER(ctypes.c_char_p)),
        ('stop_count', ctypes.c_int32),
        ('n_past', ctypes.c_int32),
        ('sampler_config', ctypes.POINTER(ml_SamplerConfig)),
        ('image_paths', ctypes.POINTER(ctypes.c_char_p)),
        ('image_count', ctypes.c_int32),
        ('image_max_length', ctypes.c_int32),
        ('audio_paths', ctypes.POINTER(ctypes.c_char_p)),
        ('audio_count', ctypes.c_int32),
    ]


class ml_ModelConfig(ctypes.Structure):
    """LLM / VLM model configuration."""

    _fields_ = [
        ('n_ctx', ctypes.c_int32),
        ('n_threads', ctypes.c_int32),
        ('n_threads_batch', ctypes.c_int32),
        ('n_batch', ctypes.c_int32),
        ('n_ubatch', ctypes.c_int32),
        ('n_seq_max', ctypes.c_int32),
        ('n_gpu_layers', ctypes.c_int32),
        ('chat_template_path', ctypes.c_char_p),
        ('chat_template_content', ctypes.c_char_p),
        ('system_prompt', ctypes.c_char_p),
        ('enable_sampling', ctypes.c_bool),
        ('grammar_str', ctypes.c_char_p),
        ('max_tokens', ctypes.c_int32),
        ('enable_thinking', ctypes.c_bool),
        ('verbose', ctypes.c_bool),
        ('qnn_model_folder_path', ctypes.c_char_p),
        ('qnn_lib_folder_path', ctypes.c_char_p),
    ]


class ml_LLM(ctypes.Structure):
    """Opaque LLM handle."""

    pass


class ml_LlmCreateInput(ctypes.Structure):
    """Input structure for creating an LLM instance."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('tokenizer_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
        ('license_id', ctypes.c_char_p),
        ('license_key', ctypes.c_char_p),
    ]


class ml_LlmChatMessage(ctypes.Structure):
    """Chat message structure."""

    _fields_ = [
        ('role', ctypes.c_char_p),
        ('content', ctypes.c_char_p),
    ]


class ml_LlmApplyChatTemplateInput(ctypes.Structure):
    """Input structure for applying chat template."""

    _fields_ = [
        ('messages', ctypes.POINTER(ml_LlmChatMessage)),
        ('message_count', ctypes.c_int32),
        ('tools', ctypes.c_char_p),
        ('enable_thinking', ctypes.c_bool),
        ('add_generation_prompt', ctypes.c_bool),
    ]


class ml_LlmApplyChatTemplateOutput(ctypes.Structure):
    """Output structure for applying chat template."""

    _fields_ = [
        ('formatted_text', ctypes.c_void_p),
    ]


class ml_ProfileData(ctypes.Structure):
    """Profile data structure for performance metrics."""

    _fields_ = [
        ('ttft', ctypes.c_int64),
        ('prompt_time', ctypes.c_int64),
        ('decode_time', ctypes.c_int64),
        ('prompt_tokens', ctypes.c_int64),
        ('generated_tokens', ctypes.c_int64),
        ('audio_duration', ctypes.c_int64),
        ('prefill_speed', ctypes.c_double),
        ('decoding_speed', ctypes.c_double),
        ('real_time_factor', ctypes.c_double),
        ('stop_reason', ctypes.c_char_p),
    ]


class ml_LlmGenerateInput(ctypes.Structure):
    """Input structure for streaming text generation."""

    _fields_ = [
        ('prompt_utf8', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_GenerationConfig)),
        ('on_token', ml_token_callback),
        ('user_data', ctypes.c_void_p),
    ]


class ml_LlmGenerateOutput(ctypes.Structure):
    """Output structure for streaming text generation."""

    _fields_ = [
        ('full_text', ctypes.c_void_p),
        ('profile_data', ml_ProfileData),
    ]


class ml_KvCacheSaveInput(ctypes.Structure):
    """Input structure for saving KV cache."""

    _fields_ = [
        ('path', ctypes.c_char_p),
    ]


class ml_KvCacheSaveOutput(ctypes.Structure):
    """Output structure for saving KV cache."""

    _fields_ = [
        ('reserved', ctypes.c_void_p),
    ]


class ml_KvCacheLoadInput(ctypes.Structure):
    """Input structure for loading KV cache."""

    _fields_ = [
        ('path', ctypes.c_char_p),
    ]


class ml_KvCacheLoadOutput(ctypes.Structure):
    """Output structure for loading KV cache."""

    _fields_ = [
        ('reserved', ctypes.c_void_p),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_llm_create.argtypes = [ctypes.POINTER(ml_LlmCreateInput), ctypes.POINTER(ctypes.POINTER(ml_LLM))]
_lib.ml_llm_create.restype = ctypes.c_int32

_lib.ml_llm_destroy.argtypes = [ctypes.POINTER(ml_LLM)]
_lib.ml_llm_destroy.restype = ctypes.c_int32

_lib.ml_llm_reset.argtypes = [ctypes.POINTER(ml_LLM)]
_lib.ml_llm_reset.restype = ctypes.c_int32

_lib.ml_llm_save_kv_cache.argtypes = [
    ctypes.POINTER(ml_LLM),
    ctypes.POINTER(ml_KvCacheSaveInput),
    ctypes.POINTER(ml_KvCacheSaveOutput),
]
_lib.ml_llm_save_kv_cache.restype = ctypes.c_int32

_lib.ml_llm_load_kv_cache.argtypes = [
    ctypes.POINTER(ml_LLM),
    ctypes.POINTER(ml_KvCacheLoadInput),
    ctypes.POINTER(ml_KvCacheLoadOutput),
]
_lib.ml_llm_load_kv_cache.restype = ctypes.c_int32

_lib.ml_llm_apply_chat_template.argtypes = [
    ctypes.POINTER(ml_LLM),
    ctypes.POINTER(ml_LlmApplyChatTemplateInput),
    ctypes.POINTER(ml_LlmApplyChatTemplateOutput),
]
_lib.ml_llm_apply_chat_template.restype = ctypes.c_int32

_lib.ml_llm_generate.argtypes = [
    ctypes.POINTER(ml_LLM),
    ctypes.POINTER(ml_LlmGenerateInput),
    ctypes.POINTER(ml_LlmGenerateOutput),
]
_lib.ml_llm_generate.restype = ctypes.c_int32
