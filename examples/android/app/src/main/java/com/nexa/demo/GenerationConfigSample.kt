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

package com.nexa.demo

import com.geniex.sdk.bean.GenerationConfig
import com.geniex.sdk.bean.SamplerConfig

// Configuration sample for generation with defaults compatible with bridge
// maxTokens: 0 = no limit, generates until model's natural stopping point
data class GenerationConfigSample(
    var maxTokens: Int = 2048,
    var stopWords: List<String>? = null,
    var stopCount: Int = 0,
    var nPast: Int = 0,
    var imagePaths: List<String>? = null,
    var imageCount: Int = 0,
    var audioPaths: List<String>? = null,
    var audioCount: Int = 0,
    var systemPrompt: String = ""
) {
    // Convert to GenerationConfig with minimal sampler setup for bridge compatibility
    // Sampler config uses bridge defaults (no custom parameters applied)
    fun toGenerationConfig(grammarString:String? = null): GenerationConfig {
        return GenerationConfig(
            maxTokens = this.maxTokens,
            stopWords = this.stopWords?.toTypedArray(),
            stopCount = this.stopCount,
            nPast = this.nPast,
            //samplerConfig = SamplerConfig(
            //    grammarString = grammarString
                // All other sampler parameters use bridge defaults
                // No temperature, topK, topP, penalties applied
            //),
            imagePaths = this.imagePaths?.toTypedArray(),
            imageCount = this.imageCount,
            audioPaths = this.audioPaths?.toTypedArray(),
            audioCount = this.audioCount
        )
    }
}