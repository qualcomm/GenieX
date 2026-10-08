// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "vlm_prefix.h"

#include <stdexcept>

using geniex::match_vlm_prefix;
using geniex::VlmPrefixChunk;

static VlmPrefixChunk text(std::initializer_list<int32_t> tokens, int32_t begin) {
    return {VlmPrefixChunk::Kind::Text, tokens, "", begin, begin + static_cast<int32_t>(tokens.size())};
}

static VlmPrefixChunk media(VlmPrefixChunk::Kind kind, const char* id, int32_t begin, int32_t end) {
    return {kind, {}, id, begin, end};
}

static void check(bool condition) {
    if (!condition) throw std::runtime_error("VLM prefix regression");
}

int main() {
    using Kind = VlmPrefixChunk::Kind;
    const std::vector<VlmPrefixChunk> previous{text({1, 2}, 0),
        media(Kind::Image, "image-a", 2, 17),
        text({3, 4, 5}, 17),
        media(Kind::Audio, "audio-a", 20, 32),
        text({6, 7}, 32)};

    auto same = match_vlm_prefix(previous, previous);
    check(same.next_chunk == 5 && same.kv_pos == 34);

    auto appended = previous;
    appended.back().tokens.push_back(8);
    auto append_match = match_vlm_prefix(previous, appended);
    check(append_match.next_chunk == 4 && append_match.text_offset == 2 && append_match.kv_pos == 34);

    auto edited      = previous;
    edited[2].tokens = {3, 9};
    auto edit_match  = match_vlm_prefix(previous, edited);
    check(edit_match.next_chunk == 2 && edit_match.text_offset == 1 && edit_match.kv_pos == 18);

    auto replaced        = previous;
    replaced[1].media_id = "image-b";
    auto replace_match   = match_vlm_prefix(previous, replaced);
    check(replace_match.next_chunk == 1 && replace_match.kv_pos == 2);

    auto reordered     = previous;
    reordered[1]       = media(Kind::Audio, "audio-a", 2, 14);
    auto reorder_match = match_vlm_prefix(previous, reordered);
    check(reorder_match.next_chunk == 1 && reorder_match.kv_pos == 2);

    auto trimmed = previous;
    trimmed.resize(3);
    auto trim_match = match_vlm_prefix(previous, trimmed);
    check(trim_match.next_chunk == 3 && trim_match.kv_pos == 20);

    auto missing = previous;
    missing[1].media_id.clear();
    check(match_vlm_prefix(previous, missing).kv_pos == 2);

    auto audio_replaced        = previous;
    audio_replaced[3].media_id = "audio-b";
    check(match_vlm_prefix(previous, audio_replaced).kv_pos == 20);
    audio_replaced[3].media_id.clear();
    check(match_vlm_prefix(previous, audio_replaced).kv_pos == 20);

    auto modality_changed    = previous;
    modality_changed[1].kind = Kind::Audio;
    check(match_vlm_prefix(previous, modality_changed).kv_pos == 2);

    auto text_trimmed = previous;
    text_trimmed.back().tokens.pop_back();
    const auto trimmed_boundary = geniex::vlm_prefill_boundary(previous, text_trimmed);
    check(trimmed_boundary.next_chunk == 4 && trimmed_boundary.text_offset == 0 && trimmed_boundary.kv_pos == 32);

    auto identical_boundary = geniex::vlm_prefill_boundary(previous, previous);
    check(identical_boundary.next_chunk == 4 && identical_boundary.text_offset == 1 && identical_boundary.kv_pos == 33);

    auto end_at_media = previous;
    end_at_media.resize(2);
    const auto media_boundary = geniex::vlm_prefill_boundary(previous, end_at_media);
    check(media_boundary.next_chunk == 1 && media_boundary.kv_pos == 2);

    auto before_media         = previous;
    before_media[0].tokens[1] = 99;
    check(match_vlm_prefix(previous, before_media).kv_pos == 1);
    check(match_vlm_prefix({}, previous).kv_pos == 0);
    check(match_vlm_prefix(previous, {}).kv_pos == 0);
    check(geniex::vlm_prefill_boundary({}, {}).kv_pos == 0);
}
