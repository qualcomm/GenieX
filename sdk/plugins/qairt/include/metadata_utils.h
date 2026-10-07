// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>

#include "utils/detail/json.hpp"

namespace geniex::qairt {

struct ChatTemplateMetadata {
    std::string default_system_prompt;
};

inline bool has_basic_genie_dialog(const std::filesystem::path& model_dir) {
    std::ifstream file(model_dir / "genie_config.json");
    if (!file) return false;

    try {
        const auto config = qualla::json::parse(file);
        return config.at("dialog").at("type") == "basic";
    } catch (const std::exception&) {
        return false;
    }
}

inline bool ensure_legacy_tokenizer_config(const std::filesystem::path& model_dir) {
    namespace fs           = std::filesystem;
    const auto config_path = model_dir / "tokenizer_config.json";
    if (fs::exists(config_path)) return true;

    std::ifstream manifest_file(model_dir / "geniex.json");
    if (!manifest_file) return false;
    try {
        const auto manifest = qualla::json::parse(manifest_file);
        const auto name     = manifest.value("ModelName", std::string{});
        if (name != "phi_3_5_mini_instruct" && name != "Phi-3.5-Mini-Instruct") return false;

        // Phi's public Genie ZIP omits tokenizer_config.json. Match the
        // Microsoft Phi-3.5 chat template without requiring a second download.
        // Source: https://huggingface.co/microsoft/Phi-3.5-mini-instruct
        // (MIT license).
        const std::string chat_template =
            "{% for message in messages %}"
            "{% if message['role'] == 'system' and message['content'] %}"
            "{{'<|system|>\n' + message['content'] + '<|end|>\n'}}"
            "{% elif message['role'] == 'user' %}"
            "{{'<|user|>\n' + message['content'] + '<|end|>\n'}}"
            "{% elif message['role'] == 'assistant' %}"
            "{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}"
            "{% else %}{{ eos_token }}{% endif %}";
        const qualla::json config = {
            {"bos_token", "<s>"}, {"eos_token", "<|endoftext|>"}, {"chat_template", chat_template}};

        std::random_device random;
        fs::path           temp_path = config_path;
        temp_path += ".tmp." + std::to_string(random()) + "." + std::to_string(random());
        {
            std::ofstream output(temp_path, std::ios::binary | std::ios::trunc);
            if (!output) return false;
            output << config.dump();
            output.close();
            if (!output) {
                std::error_code ec;
                fs::remove(temp_path, ec);
                return false;
            }
        }
        std::error_code ec;
        fs::rename(temp_path, config_path, ec);
        if (ec) {
            fs::remove(temp_path, ec);
            return fs::exists(config_path);
        }
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

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
