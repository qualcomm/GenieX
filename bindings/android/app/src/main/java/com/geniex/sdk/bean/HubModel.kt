// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.bean

/** Mirrors `geniex_HubModelInfo`. One AI Hub model geniex can run (qairt). */
data class HubModel(
    val name: String,
    val model_type: ModelType,
    val chipsets: Array<String>,
)
