// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.bean

/**
 * A function call issued on an assistant turn. Chat templates render the
 * following tool response from these, matching it by [id], so a call flattened
 * into assistant text costs the response its place in the prompt.
 */
data class ToolCall(
    val id: String? = null,
    val name: String,
    val arguments: String
)
