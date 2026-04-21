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

from ._api import (
    GeniexError,
    _check,
    _encode,
    _str_list_to_c,
    deinit,
    ensure_init,
    get_device_list,
    get_plugin_list,
    init,
    version,
)
from ._lib import load_library
from ._types import (
    geniex_GenerationConfig,
    geniex_GetDeviceListInput,
    geniex_GetDeviceListOutput,
    geniex_GetPluginListOutput,
    geniex_KvCacheLoadInput,
    geniex_KvCacheLoadOutput,
    geniex_KvCacheSaveInput,
    geniex_KvCacheSaveOutput,
    geniex_LlmApplyChatTemplateInput,
    geniex_LlmApplyChatTemplateOutput,
    geniex_LlmChatMessage,
    geniex_LlmCreateInput,
    geniex_LlmGenerateInput,
    geniex_LlmGenerateOutput,
    geniex_ModelConfig,
    geniex_ProfileData,
    geniex_SamplerConfig,
    geniex_token_callback,
    geniex_VlmApplyChatTemplateInput,
    geniex_VlmApplyChatTemplateOutput,
    geniex_VlmChatMessage,
    geniex_VlmContent,
    geniex_VlmCreateInput,
    geniex_VlmGenerateInput,
    geniex_VlmGenerateOutput,
)

__all__ = [
    'GeniexError',
    '_check',
    '_encode',
    '_str_list_to_c',
    'deinit',
    'ensure_init',
    'get_device_list',
    'get_plugin_list',
    'init',
    'load_library',
    'version',
    'geniex_GenerationConfig',
    'geniex_GetDeviceListInput',
    'geniex_GetDeviceListOutput',
    'geniex_GetPluginListOutput',
    'geniex_KvCacheLoadInput',
    'geniex_KvCacheLoadOutput',
    'geniex_KvCacheSaveInput',
    'geniex_KvCacheSaveOutput',
    'geniex_LlmApplyChatTemplateInput',
    'geniex_LlmApplyChatTemplateOutput',
    'geniex_LlmChatMessage',
    'geniex_LlmCreateInput',
    'geniex_LlmGenerateInput',
    'geniex_LlmGenerateOutput',
    'geniex_ModelConfig',
    'geniex_ProfileData',
    'geniex_SamplerConfig',
    'geniex_VlmApplyChatTemplateInput',
    'geniex_VlmApplyChatTemplateOutput',
    'geniex_VlmChatMessage',
    'geniex_VlmContent',
    'geniex_VlmCreateInput',
    'geniex_VlmGenerateInput',
    'geniex_VlmGenerateOutput',
    'geniex_token_callback',
]
