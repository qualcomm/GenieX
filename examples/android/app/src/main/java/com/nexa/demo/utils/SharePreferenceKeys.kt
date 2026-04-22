// Copyright 2024-2026 Nexa AI, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.nexa.demo.utils

class SharePreferenceKeys {
    companion object {
        const val KEY_LAST_LOAD_MODEL_ID = "last_load_model_id"
        /**
         * When navigation pops, since it will load the initial modelId, cannot directly load model,
         * should let MainChatScreen load it
         */
        const val KEY_PREPARE_LOAD_MODEL_ID = "prepare_load_model_id"
        const val KEY_SHOW_OPERATION_POPUP = "show_operation_popup"
    }
    enum class FileName(val fileName: String) {
        ModelDownloadState("mode_download_state"),
        CommonConfig("common_config")
    }
}