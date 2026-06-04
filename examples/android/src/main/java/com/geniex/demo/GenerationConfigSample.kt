// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

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