# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ctypes struct/type definitions mirroring sdk/include/ml.h."""

from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_char_p,
    c_double,
    c_float,
    c_int32,
    c_int64,
    c_void_p,
)

# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

# bool (*ml_token_callback)(const char* token, void* user_data)
ml_token_callback = CFUNCTYPE(c_bool, c_char_p, c_void_p)

# ---------------------------------------------------------------------------
# ml_ProfileData
# ---------------------------------------------------------------------------


class ml_ProfileData(Structure):
    _fields_ = [
        ('ttft', c_int64),
        ('prompt_time', c_int64),
        ('decode_time', c_int64),
        ('prompt_tokens', c_int64),
        ('generated_tokens', c_int64),
        ('audio_duration', c_int64),
        ('prefill_speed', c_double),
        ('decoding_speed', c_double),
        ('real_time_factor', c_double),
        ('stop_reason', c_char_p),
    ]


# ---------------------------------------------------------------------------
# ml_SamplerConfig
# ---------------------------------------------------------------------------


class ml_SamplerConfig(Structure):
    _fields_ = [
        ('temperature', c_float),
        ('top_p', c_float),
        ('top_k', c_int32),
        ('min_p', c_float),
        ('repetition_penalty', c_float),
        ('presence_penalty', c_float),
        ('frequency_penalty', c_float),
        ('seed', c_int32),
        ('grammar_path', c_char_p),
        ('grammar_string', c_char_p),
        ('enable_json', c_bool),
    ]


# ---------------------------------------------------------------------------
# ml_GenerationConfig
# ---------------------------------------------------------------------------


class ml_GenerationConfig(Structure):
    _fields_ = [
        ('max_tokens', c_int32),
        ('stop', POINTER(c_char_p)),
        ('stop_count', c_int32),
        ('n_past', c_int32),
        ('sampler_config', POINTER(ml_SamplerConfig)),
        ('image_paths', POINTER(c_char_p)),
        ('image_count', c_int32),
        ('image_max_length', c_int32),
        ('audio_paths', POINTER(c_char_p)),
        ('audio_count', c_int32),
    ]


# ---------------------------------------------------------------------------
# ml_ModelConfig
# ---------------------------------------------------------------------------


class ml_ModelConfig(Structure):
    _fields_ = [
        ('n_ctx', c_int32),
        ('n_threads', c_int32),
        ('n_threads_batch', c_int32),
        ('n_batch', c_int32),
        ('n_ubatch', c_int32),
        ('n_seq_max', c_int32),
        ('n_gpu_layers', c_int32),
        ('chat_template_path', c_char_p),
        ('chat_template_content', c_char_p),
        ('system_prompt', c_char_p),
        ('enable_sampling', c_bool),
        ('grammar_str', c_char_p),
        ('max_tokens', c_int32),
        ('enable_thinking', c_bool),
        ('verbose', c_bool),
        ('qnn_model_folder_path', c_char_p),
        ('qnn_lib_folder_path', c_char_p),
    ]


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class ml_KvCacheSaveInput(Structure):
    _fields_ = [('path', c_char_p)]


class ml_KvCacheSaveOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


class ml_KvCacheLoadInput(Structure):
    _fields_ = [('path', c_char_p)]


class ml_KvCacheLoadOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


# ---------------------------------------------------------------------------
# LLM structs
# ---------------------------------------------------------------------------


class ml_LlmCreateInput(Structure):
    _fields_ = [
        ('model_name', c_char_p),
        ('model_path', c_char_p),
        ('tokenizer_path', c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('license_id', c_char_p),
        ('license_key', c_char_p),
    ]


class ml_LlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(ml_GenerationConfig)),
        ('on_token', ml_token_callback),
        ('user_data', c_void_p),
        ('input_ids', POINTER(c_int32)),
        ('input_ids_count', c_int32),
    ]


class ml_LlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', ml_ProfileData),
    ]


class ml_LlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('content', c_char_p),
    ]


class ml_LlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(ml_LlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('add_generation_prompt', c_bool),
    ]


class ml_LlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


# ---------------------------------------------------------------------------
# VLM structs
# ---------------------------------------------------------------------------


class ml_VlmContent(Structure):
    # field name in C is `type` but that's a Python keyword — use _type_ alias
    _fields_ = [
        ('type', c_char_p),
        ('text', c_char_p),
    ]


class ml_VlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('contents', POINTER(ml_VlmContent)),
        ('content_count', c_int64),  # int64_t in ml.h
    ]


class ml_VlmCreateInput(Structure):
    _fields_ = [
        ('model_name', c_char_p),
        ('model_path', c_char_p),
        ('mmproj_path', c_char_p),
        ('config', ml_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('tokenizer_path', c_char_p),
        ('license_id', c_char_p),
        ('license_key', c_char_p),
    ]


class ml_VlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(ml_GenerationConfig)),
        ('on_token', ml_token_callback),
        ('user_data', c_void_p),
    ]


class ml_VlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', ml_ProfileData),
    ]


class ml_VlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(ml_VlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('grounding', c_bool),
    ]


class ml_VlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


# ---------------------------------------------------------------------------
# Plugin / device query structs
# ---------------------------------------------------------------------------


class ml_GetPluginListOutput(Structure):
    _fields_ = [
        ('plugin_ids', POINTER(c_char_p)),
        ('plugin_count', c_int32),
    ]


class ml_GetDeviceListInput(Structure):
    _fields_ = [('plugin_id', c_char_p)]


class ml_GetDeviceListOutput(Structure):
    _fields_ = [
        ('device_ids', POINTER(c_char_p)),
        ('device_names', POINTER(c_char_p)),
        ('device_count', c_int32),
    ]
