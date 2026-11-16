// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "decision.h"

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <stdexcept>

#include "chat-auto-parser.h"
#include "chat.h"
#include "common.h"
#include "jinja/runtime.h"
#include "liquid-decision.h"
#include "logging.h"
#include "mtmd.h"

namespace geniex {

std::string decision_metadata_string(const llama_model* model, const std::string& key) {
    char value[256] = {};
    return llama_model_meta_val_str(model, key.c_str(), value, sizeof(value)) < 0 ? "" : value;
}

namespace {

using json = common_json;

// Keep request values as C++ types until building the Jinja context.
struct Option {
    std::string                key;
    std::optional<std::string> description;
    bool                       key_is_bool;
};
struct Question {
    std::string         id;
    std::string         type;
    std::string         instructions;
    std::vector<Option> options;
};

// Reserve mtmd's marker for media placeholders inserted by the SDK.
std::string clean_marker(std::string text, const std::string& marker) {
    if (marker.empty()) return text;
    for (size_t pos = 0; (pos = text.find(marker, pos)) != std::string::npos;) {
        text.replace(pos, marker.size(), " ");
        ++pos;
    }
    return text;
}

// Validate caller-owned C arrays and normalize public types for systemone.
std::vector<Question> parse_questions(const geniex_DecideInput* input, const std::string& marker) {
    std::vector<Question> result;
    result.reserve(input->question_count);
    for (int32_t i = 0; i < input->question_count; ++i) {
        const auto& item = input->questions[i];
        if (!item.question_type || !item.instructions || item.option_count <= 0 || !item.options) {
            throw std::invalid_argument("decision questions require a type, instructions, and options");
        }
        Question question;
        question.id   = clean_marker(item.id && *item.id ? item.id : "question_" + std::to_string(i), marker);
        question.type = std::string(item.question_type) == "predicate" ? "noul" : item.question_type;
        if (question.type != "choice" && question.type != "score" && question.type != "noul") {
            throw std::invalid_argument("unsupported decision question type");
        }
        question.instructions = clean_marker(item.instructions, marker);
        for (int32_t j = 0; j < item.option_count; ++j) {
            const auto& option = item.options[j];
            if (!option.key ||
                (option.key_is_bool && std::strcmp(option.key, "true") != 0 && std::strcmp(option.key, "false") != 0)) {
                throw std::invalid_argument("decision options require valid keys");
            }
            question.options.push_back({clean_marker(option.key, marker),
                option.description ? std::optional(clean_marker(option.description, marker)) : std::nullopt,
                option.key_is_bool});
        }
        if (question.type == "noul" &&
            (question.options.size() != 2 || question.options[0].key_is_bool || question.options[1].key_is_bool ||
                !((question.options[0].key == "true" && question.options[1].key == "false") ||
                    (question.options[0].key == "false" && question.options[1].key == "true")))) {
            throw std::invalid_argument("noul questions require false and true options");
        }
        result.push_back(std::move(question));
    }
    return result;
}

// Preserve boolean keys and absent descriptions when entering Jinja.
json option_json(const Option& option, const std::string& label, bool include_label) {
    json result = json{{"key", option.key_is_bool ? json(option.key == "true") : json(option.key)},
        {"description", option.description ? json(*option.description) : json()}};
    if (include_label) result["label"] = label;
    return result;
}

// mtmd binds media to placeholders in prompt order, so counts must match.
size_t marker_count(const std::string& text, const std::string& marker) {
    size_t count = 0;
    for (size_t pos = 0; (pos = text.find(marker, pos)) != std::string::npos; pos += marker.size()) ++count;
    return count;
}

}  // namespace

int32_t prepare_decision(const llama_model* model, const geniex_DecideInput* input, DecisionPrepared* output) {
    if (!model || !input || !output || !input->state || !input->questions || input->question_count <= 0 ||
        input->question_index < 0 || input->question_index >= input->question_count || input->image_count < 0 ||
        input->audio_count < 0 || (input->image_count > 0 && !input->image_paths) ||
        (input->audio_count > 0 && !input->audio_paths)) {
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    for (int32_t i = 0; i < input->image_count; ++i) {
        if (!input->image_paths[i]) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
    for (int32_t i = 0; i < input->audio_count; ++i) {
        if (!input->audio_paths[i]) return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    try {
        const std::string architecture  = decision_metadata_string(model, "general.architecture");
        const std::string metadata_type = decision_metadata_string(model, architecture + ".decision.type");
        if (!metadata_type.empty() && metadata_type != "openjev" && metadata_type != "lev" &&
            metadata_type != "nimble" && !is_liquid_decision(metadata_type)) {
            return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
        }
        // Use the model's native template when present; otherwise render a generic chat prompt.
        const char* source = llama_model_chat_template(model, "systemone");
        // A declared Liquid protocol cannot be answered by generic chat prompts.
        if (is_liquid_decision(metadata_type) && !source) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
        const std::string type = !source ? "generic" : metadata_type.empty() ? "openjev" : metadata_type;
        // Liquid text never goes through mtmd. Its own escaping rules must see
        // the original caller text, including a literal SDK media marker.
        const std::string marker    = is_liquid_decision(type) ? "" : mtmd_default_marker();
        const std::string state     = clean_marker(input->state, marker);
        const auto        questions = parse_questions(input, marker);

        if (is_liquid_decision(type)) {
            // Media needs a different batch assembly path, particularly for Omni.
            // Reject it before rendering instead of dropping it from the state.
            if (input->image_count || input->audio_count) return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
            const auto&      question = questions[input->question_index];
            DecisionPrepared prepared;
            prepared.type          = type;
            prepared.question_type = question.type;
            json options           = json::array();
            for (size_t i = 0; i < question.options.size(); ++i) prepared.original_index.push_back(i);
            const bool true_first = type != "lfm2-d1-omni";
            if (question.type == "noul" && question.options[0].key != (true_first ? "true" : "false")) {
                std::reverse(prepared.original_index.begin(), prepared.original_index.end());
            }
            for (const size_t i : prepared.original_index) {
                const auto& option = question.options[i];
                // Score levels are positional, regardless of the caller's display keys.
                options.push_back(json{{"key", question.type == "score" ? std::to_string(i) : option.key},
                    {"description", option.description ? json(*option.description) : json()}});
            }
            prepare_liquid_decision(model,
                json{{"id", question.id},
                    {"type", question.type},
                    {"state", state},
                    {"instructions", question.instructions},
                    {"options", options}},
                prepared);
            *output = std::move(prepared);
            return GENIEX_SUCCESS;
        }

        // Only single-token answer codes can be read directly from next-token logits.
        std::vector<std::string> labels;
        const llama_vocab*       vocab = llama_model_get_vocab(model);
        if (type == "openjev") {
            for (char label = 'A'; label <= 'Z'; ++label) {
                const auto tokens = common_tokenize(vocab, std::string(1, label), false, false);
                if (tokens.size() != 1) throw std::invalid_argument("openjev label is not a single token");
                labels.push_back(std::string(1, label));
            }
            for (char label = 'a'; label <= 'z'; ++label) {
                const auto tokens = common_tokenize(vocab, std::string(1, label), false, false);
                if (tokens.size() != 1) throw std::invalid_argument("openjev label is not a single token");
                labels.push_back(std::string(1, label));
            }
        } else {
            for (char a = 'A'; a <= 'Z'; ++a) labels.emplace_back(1, a);
            for (char a = 'A'; a <= 'Z'; ++a) {
                for (char b = 'A'; b <= 'Z'; ++b) labels.push_back(std::string{a, b});
            }
            std::vector<std::string> single_token_labels;
            for (const auto& label : labels) {
                if (single_token_labels.size() >= 255) break;
                if (common_tokenize(vocab, label, false, false).size() == 1) single_token_labels.push_back(label);
            }
            labels = std::move(single_token_labels);
        }

        // Render noul in model order while retaining caller indices for the answer.
        const Question&     original = questions[input->question_index];
        std::vector<size_t> order(original.options.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        if (original.type == "noul" && type != "generic" &&
            original.options[0].key == (type == "openjev" ? "false" : "true")) {
            std::reverse(order.begin(), order.end());
        }
        if (order.size() > labels.size()) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        if (type == "nimble" && std::any_of(questions.begin(), questions.end(), [&labels](const Question& question) {
                return question.options.size() > labels.size();
            })) {
            return GENIEX_ERROR_COMMON_INVALID_INPUT;
        }

        std::string prompt;
        std::string reversed_prompt;
        if (type == "generic") {
            // Ordinary models get explicit options inside their usual chat template.
            std::string user;
            for (int32_t i = 0; i < input->image_count + input->audio_count; ++i) {
                user += marker + "\n";
            }
            user += "State:\n" + state + "\n\nQuestion:\n" + original.instructions + "\n\nOptions:\n";
            for (size_t i = 0; i < order.size(); ++i) {
                const auto& option = original.options[order[i]];
                user += labels[i] + ". " + option.key;
                if (option.description) user += ": " + *option.description;
                user += "\n";
            }
            user += "\nReply with exactly one option label.";

            if (llama_model_chat_template(model, nullptr)) {
                common_chat_templates_ptr    tmpls = common_chat_templates_init(model, "");
                common_chat_templates_inputs request;
                common_chat_msg              msg;
                msg.role    = "user";
                msg.content = user;
                request.messages.push_back(std::move(msg));
                request.use_jinja             = true;
                request.add_generation_prompt = true;
                request.enable_thinking       = false;
                prompt                        = common_chat_templates_apply(tmpls.get(), request).prompt.str();
            } else {
                prompt = "User:\n" + user + "\nAssistant:\n";
            }
        } else {
            // Native templates receive structured values, not a pre-serialized request.
            auto render_options = [&labels, &type](const Question& question, const std::vector<size_t>& indices) {
                json options = json::array();
                for (size_t position = 0; position < indices.size(); ++position) {
                    const size_t index = indices[position];
                    options.push_back(
                        option_json(question.options[index], labels[position], type == "nimble" || type == "lev"));
                }
                return options;
            };
            common_chat_template tmpl(source, "", "");
            auto                 render = [&](const std::vector<size_t>& indices) {
                json target{{"id", original.id},
                                    {"type", original.type},
                                    {"instructions", original.instructions},
                                    {"state", state},
                                    {"options", render_options(original, indices)}};
                if (type == "nimble") {
                    // Nimble needs the complete schema, not just the selected question.
                    target["questions"] = json::array();
                    for (const auto& question : questions) {
                        std::vector<size_t> question_indices(question.options.size());
                        for (size_t i = 0; i < question_indices.size(); ++i) question_indices[i] = i;
                        if (question.type == "noul" && question.options[0].key == "true") {
                            std::reverse(question_indices.begin(), question_indices.end());
                        }
                        json all{{"id", question.id},
                                            {"type", question.type},
                                            {"instructions", question.instructions},
                                            {"options", render_options(question, question_indices)}};
                        target["questions"].push_back(std::move(all));
                    }
                }
                target["images"] = json::array();
                target["audios"] = json::array();
                for (int32_t i = 0; i < input->image_count; ++i) target["images"].push_back(marker);
                for (int32_t i = 0; i < input->audio_count; ++i) target["audios"].push_back(marker);
                autoparser::generation_params params;
                params.extra_context = std::move(target);
                params.mark_input    = false;
                return common_chat_template_direct_apply(tmpl, params);
            };
            prompt = render(order).str();
            // LEV averages calibrated choice probabilities from both option orders.
            if (type == "lev" && original.type == "choice" && order.size() > 1) {
                const std::vector<size_t> reversed(order.rbegin(), order.rend());
                reversed_prompt = render(reversed).str();
            }
        }
        // Catch templates that omit or duplicate the inserted media placeholders.
        const size_t expected_media = static_cast<size_t>(input->image_count + input->audio_count);
        if (marker_count(prompt, marker) != expected_media ||
            (!reversed_prompt.empty() && marker_count(reversed_prompt, marker) != expected_media)) {
            throw std::invalid_argument("decision prompt emitted an unexpected number of media markers");
        }

        // Prefer bucket-specific GGUF calibration, then the question-type default.
        const size_t n_options   = original.options.size();
        float        temperature = 1.0f;
        if (type != "generic") {
            const std::string bucket = type == "lev" ? (n_options <= 8       ? "small"
                                                           : n_options <= 26 ? "mid"
                                                                             : "large")
                                                     : (n_options <= 2       ? "2"
                                                           : n_options <= 5  ? "3_5"
                                                           : n_options <= 10 ? "6_10"
                                                                             : "11");
            const std::string prefix = architecture + ".decision.temperature.";
            for (const auto& key : {original.type + "." + bucket, original.type}) {
                char value[64] = {};
                if (llama_model_meta_val_str(model, (prefix + key).c_str(), value, sizeof(value)) < 0) continue;
                temperature = std::strtof(value, nullptr);
                if (temperature <= 0.0f) throw std::invalid_argument("invalid decision temperature metadata");
                break;
            }
        }

        // Carry the rendering order and answer codes through the LLM/VLM logits pass.
        output->prompt          = std::move(prompt);
        output->reversed_prompt = std::move(reversed_prompt);
        output->type            = type;
        output->question_type   = original.type;
        output->temperature     = temperature;
        output->original_index  = std::move(order);
        if (original.type == "noul") {
            for (size_t i = 0; i < n_options; ++i) {
                if (original.options[i].key == "true") output->true_index = i;
            }
        }
        const size_t n_outputs = type == "lev" && original.type == "noul" ? 9 : n_options;
        if (n_outputs > labels.size()) return GENIEX_ERROR_COMMON_INVALID_INPUT;
        output->candidates.assign(labels.begin(), labels.begin() + n_outputs);
        return GENIEX_SUCCESS;
    } catch (const std::exception& error) {
        GENIEX_LOG_WARN("decision request rejected: {}", error.what());
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
}

}  // namespace geniex
