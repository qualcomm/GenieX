// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace geniex {

struct VlmPrefixChunk {
    enum class Kind { Text, Image, Audio } kind;
    std::vector<int32_t> tokens;
    std::string          media_id;
    int32_t              kv_begin = 0;
    int32_t              kv_end   = 0;
};

struct VlmPrefixMatch {
    size_t  next_chunk  = 0;
    size_t  text_offset = 0;
    int32_t kv_pos      = 0;
};

inline VlmPrefixMatch match_vlm_prefix(
    const std::vector<VlmPrefixChunk>& previous, const std::vector<VlmPrefixChunk>& current) {
    VlmPrefixMatch match;
    const size_t   n = std::min(previous.size(), current.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& old = previous[i];
        const auto& now = current[i];
        if (old.kind != now.kind) break;
        if (old.kind != VlmPrefixChunk::Kind::Text) {
            if (old.media_id.empty() || old.media_id != now.media_id) break;
            match = {i + 1, 0, old.kv_end};
            continue;
        }
        const size_t common =
            std::mismatch(old.tokens.begin(), old.tokens.end(), now.tokens.begin(), now.tokens.end()).first -
            old.tokens.begin();
        if (common < old.tokens.size() || common < now.tokens.size()) {
            match = {i, common, old.kv_begin + static_cast<int32_t>(common)};
            break;
        }
        match = {i + 1, 0, old.kv_end};
    }
    return match;
}

inline VlmPrefixMatch vlm_prefill_boundary(
    const std::vector<VlmPrefixChunk>& previous, const std::vector<VlmPrefixChunk>& current) {
    auto match = match_vlm_prefix(previous, current);
    if (current.empty()) return match;
    const size_t last = current.size() - 1;
    if (match.next_chunk == current.size()) {
        match = {last, current[last].tokens.size(), previous[last].kv_end};
        if (current[last].kind != VlmPrefixChunk::Kind::Text) {
            return {last, 0, previous[last].kv_begin};
        }
    }
    // Re-evaluate the last token to restore logits after truncation.
    if (match.next_chunk == last && current[last].kind == VlmPrefixChunk::Kind::Text &&
        match.text_offset == current[last].tokens.size() && match.text_offset > 0) {
        --match.text_offset;
        --match.kv_pos;
    }
    return match;
}

}  // namespace geniex
