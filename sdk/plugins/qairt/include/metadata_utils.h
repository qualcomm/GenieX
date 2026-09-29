// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#include "utils/detail/json.hpp"

namespace geniex::qairt {

struct ChatTemplateMetadata {
    std::string default_system_prompt;
};

inline ChatTemplateMetadata read_chat_template_metadata(const std::filesystem::path& model_dir) {
    ChatTemplateMetadata result;
    const auto           metadata_path = model_dir / "metadata.json";
    std::ifstream        file(metadata_path);
    if (!file) return result;

    try {
        const auto  metadata = qualla::json::parse(file);
        const auto& prompt   = metadata.at("genie").at("chat_template").at("default_system_prompt");
        if (prompt.is_string()) result.default_system_prompt = prompt.get<std::string>();
    } catch (const std::exception&) {
    }
    return result;
}

}  // namespace geniex::qairt
