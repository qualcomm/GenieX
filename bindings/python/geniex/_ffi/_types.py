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

"""ctypes struct/type definitions mirroring sdk/include/geniex.h."""

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
    c_uint32,
    c_void_p,
)

# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------

# bool (*geniex_token_callback)(const char* token, void* user_data)
geniex_token_callback = CFUNCTYPE(c_bool, c_char_p, c_void_p)

# ---------------------------------------------------------------------------
# geniex_ProfileData
# ---------------------------------------------------------------------------


class geniex_ProfileData(Structure):
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
# geniex_SamplerConfig
# ---------------------------------------------------------------------------


class geniex_SamplerConfig(Structure):
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
# geniex_GenerationConfig
# ---------------------------------------------------------------------------


class geniex_GenerationConfig(Structure):
    _fields_ = [
        ('max_tokens', c_int32),
        ('stop', POINTER(c_char_p)),
        ('stop_count', c_int32),
        ('n_past', c_int32),
        ('sampler_config', POINTER(geniex_SamplerConfig)),
        ('image_paths', POINTER(c_char_p)),
        ('image_count', c_int32),
        ('image_max_length', c_int32),
        ('audio_paths', POINTER(c_char_p)),
        ('audio_count', c_int32),
    ]


# ---------------------------------------------------------------------------
# geniex_ModelConfig
# ---------------------------------------------------------------------------


class geniex_ModelConfig(Structure):
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
    ]


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class geniex_KvCacheSaveInput(Structure):
    _fields_ = [('path', c_char_p)]


class geniex_KvCacheSaveOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


class geniex_KvCacheLoadInput(Structure):
    _fields_ = [('path', c_char_p)]


class geniex_KvCacheLoadOutput(Structure):
    _fields_ = [('reserved', c_void_p)]


# ---------------------------------------------------------------------------
# LLM structs
# ---------------------------------------------------------------------------


class geniex_LlmCreateInput(Structure):
    _fields_ = [
        ('model_name', c_char_p),
        ('model_path', c_char_p),
        ('tokenizer_path', c_char_p),
        ('config', geniex_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('license_id', c_char_p),
        ('license_key', c_char_p),
    ]


class geniex_LlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(geniex_GenerationConfig)),
        ('on_token', geniex_token_callback),
        ('user_data', c_void_p),
        ('input_ids', POINTER(c_int32)),
        ('input_ids_count', c_int32),
    ]


class geniex_LlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', geniex_ProfileData),
    ]


class geniex_LlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('content', c_char_p),
    ]


class geniex_LlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(geniex_LlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('add_generation_prompt', c_bool),
    ]


class geniex_LlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


# ---------------------------------------------------------------------------
# VLM structs
# ---------------------------------------------------------------------------


class geniex_VlmContent(Structure):
    _fields_ = [
        ('type', c_char_p),
        ('text', c_char_p),
    ]


class geniex_VlmChatMessage(Structure):
    _fields_ = [
        ('role', c_char_p),
        ('contents', POINTER(geniex_VlmContent)),
        ('content_count', c_int64),  # int64_t in geniex.h
    ]


class geniex_VlmCreateInput(Structure):
    _fields_ = [
        ('model_name', c_char_p),
        ('model_path', c_char_p),
        ('mmproj_path', c_char_p),
        ('config', geniex_ModelConfig),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
        ('tokenizer_path', c_char_p),
        ('license_id', c_char_p),
        ('license_key', c_char_p),
    ]


class geniex_VlmGenerateInput(Structure):
    _fields_ = [
        ('prompt_utf8', c_char_p),
        ('config', POINTER(geniex_GenerationConfig)),
        ('on_token', geniex_token_callback),
        ('user_data', c_void_p),
    ]


class geniex_VlmGenerateOutput(Structure):
    _fields_ = [
        ('full_text', c_void_p),
        ('profile_data', geniex_ProfileData),
    ]


class geniex_VlmApplyChatTemplateInput(Structure):
    _fields_ = [
        ('messages', POINTER(geniex_VlmChatMessage)),
        ('message_count', c_int32),
        ('tools', c_char_p),
        ('enable_thinking', c_bool),
        ('grounding', c_bool),
    ]


class geniex_VlmApplyChatTemplateOutput(Structure):
    _fields_ = [('formatted_text', c_void_p)]


class geniex_VlmCapabilities(Structure):
    _fields_ = [
        ('supports_vision', c_bool),
        ('supports_audio', c_bool),
    ]


# ---------------------------------------------------------------------------
# Plugin / device query structs
# ---------------------------------------------------------------------------


class geniex_GetPluginListOutput(Structure):
    _fields_ = [
        ('plugin_ids', POINTER(c_char_p)),
        ('plugin_count', c_int32),
    ]


class geniex_GetDeviceListInput(Structure):
    _fields_ = [('plugin_id', c_char_p)]


class geniex_GetDeviceListOutput(Structure):
    _fields_ = [
        ('device_ids', POINTER(c_char_p)),
        ('device_names', POINTER(c_char_p)),
        ('device_count', c_int32),
    ]


class geniex_ResolveDeviceInput(Structure):
    _fields_ = [
        ('plugin_id', c_char_p),
        ('model_name', c_char_p),
        ('mode', c_char_p),
        ('ngl_default', c_int32),
    ]


class geniex_ResolveDeviceOutput(Structure):
    _fields_ = [
        ('device_id', c_void_p),  # heap char*, caller frees with geniex_free
        ('ngl', c_int32),
        ('warning', c_void_p),  # heap char*, caller frees with geniex_free
    ]


# ---------------------------------------------------------------------------
# Model-manager structs (mirrors sdk/model-manager/include/geniex_model.h)
# ---------------------------------------------------------------------------

# geniex_ModelType enum
GENIEX_MODEL_TYPE_LLM = 0
GENIEX_MODEL_TYPE_VLM = 1

# geniex_HubSource enum
GENIEX_HUB_AUTO = 0
GENIEX_HUB_HUGGINGFACE = 1
GENIEX_HUB_MODELSCOPE = 2
GENIEX_HUB_AIHUB = 3
GENIEX_HUB_VOLCES = 4
GENIEX_HUB_LOCALFS = 127


class geniex_ModelPaths(Structure):
    _fields_ = [
        ('model_path', c_char_p),
        ('mmproj_path', c_char_p),
        ('tokenizer_path', c_char_p),
        ('model_dir', c_char_p),
        ('model_name', c_char_p),
        ('plugin_id', c_char_p),
        ('device_id', c_char_p),
    ]


class geniex_ModelListOutput(Structure):
    _fields_ = [
        ('names', POINTER(c_char_p)),
        ('count', c_int32),
    ]


class geniex_FileProgress(Structure):
    _fields_ = [
        ('file_name', c_char_p),
        ('downloaded_bytes', c_int64),
        ('total_bytes', c_int64),
    ]


# bool (*)(const geniex_FileProgress* files, int32_t file_count, void* user_data)
geniex_download_progress_cb = CFUNCTYPE(c_bool, POINTER(geniex_FileProgress), c_int32, c_void_p)


class geniex_ModelPullInput(Structure):
    _fields_ = [
        # ABI version gate. Must equal ctypes.sizeof(geniex_ModelPullInput)
        # at the call site — the Rust side rejects any other value with
        # GENIEX_ERROR_COMMON_INVALID_INPUT. Our pull() wrapper sets it
        # automatically; external callers using this struct directly must
        # set it themselves.
        ('struct_size', c_uint32),
        ('model_name', c_char_p),
        ('quant', c_char_p),
        ('hub', c_int32),
        ('local_path', c_char_p),
        ('hf_token', c_char_p),
        ('chipset', c_char_p),
        ('display_name', c_char_p),
        ('on_progress', geniex_download_progress_cb),
        ('user_data', c_void_p),
    ]
