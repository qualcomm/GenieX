// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.bean

data class DecideOption(
    val key: String,
    val keyIsBool: Boolean = false,
    val description: String? = null,
)

data class DecideQuestion(
    val id: String,
    val questionType: String,
    val instructions: String,
    val options: Array<DecideOption>,
)

data class DecideInput(
    val state: String,
    val questions: Array<DecideQuestion>,
    val questionIndex: Int = 0,
    val imagePaths: Array<String> = emptyArray(),
    val audioPaths: Array<String> = emptyArray(),
)

data class DecideOutput(
    val logits: FloatArray? = null,
    val probabilities: FloatArray? = null,
    val inputTokens: Int = 0,
    val temperature: Float = 1f,
    val decisionType: String = "",
)
