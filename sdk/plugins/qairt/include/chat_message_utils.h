// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
//
// Translates the public C tool-calling fields of a chat message into qairt-core
// `geniex::ChatMessage`. Shared by the qairt LLM and VLM plugins.

#pragma once

#include <cstdint>
#include <utility>

#include "geniex-proc/types.h"  // geniex::ChatMessage, geniex::ToolCall
#include "geniex.h"
#include "logging.h"

namespace geniex::qairt {

// Templates need tool_calls structurally, not as assistant text: most render a
// "tool" message only from the tool_calls of the assistant turn before it
// (matched by id), so dropping them drops the tool result from the prompt.
inline void apply_tool_fields(ChatMessage& msg, const geniex_ToolCall* tool_calls, int32_t tool_call_count,
    const char* tool_call_id, const char* tool_name) {
    if (tool_calls && tool_call_count > 0) {
        msg.tool_calls.reserve(static_cast<std::size_t>(tool_call_count));
        for (int32_t i = 0; i < tool_call_count; ++i) {
            const geniex_ToolCall& src = tool_calls[i];
            if (!src.name) {
                GENIEX_LOG_WARN("Skipping tool call {} with null name", i);
                continue;
            }
            ToolCall tc;
            tc.name           = src.name;
            tc.arguments_json = src.arguments ? src.arguments : "{}";
            tc.id             = src.id ? src.id : "";
            msg.tool_calls.push_back(std::move(tc));
        }
    }
    if (tool_call_id) msg.tool_call_id = tool_call_id;
    if (tool_name) msg.name = tool_name;
}

}  // namespace geniex::qairt
