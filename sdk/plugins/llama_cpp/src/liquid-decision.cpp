// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// Copyright (c) 2023-2026 The ggml authors
// SPDX-License-Identifier: BSD-3-Clause AND MIT
//
// Protocol layout and verbalizers follow llama.cpp's server-decision.cpp.
// The upstream MIT permission notice is in third-party/llama.cpp/LICENSE.

#include "liquid-decision.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "chat.h"
#include "jinja/runtime.h"
#include "logging.h"

namespace geniex {
namespace {

using json                        = common_json;
constexpr size_t  omni_max_tokens = 16384;
const std::string omni_sep        = "<<d1omni:sep>>";
const std::string omni_state      = "<<d1omni:state>>";
const std::string omni_question   = "<<d1omni:question>>";
const std::string omni_option     = "<<d1omni:option>>";

float meta_number(const llama_model* model, const std::string& key, float fallback) {
    char          text[256] = {};
    const int32_t length    = llama_model_meta_val_str(model, key.c_str(), text, sizeof(text));
    if (length < 0) return fallback;
    // Only missing fields use defaults; an empty or truncated value is corrupt.
    if (length == 0 || static_cast<size_t>(length) >= sizeof(text)) {
        throw std::invalid_argument("invalid Liquid calibration metadata: " + key);
    }
    char*       end   = nullptr;
    const float value = std::strtof(text, &end);
    if (end != text + length || !std::isfinite(value)) {
        throw std::invalid_argument("invalid Liquid calibration metadata: " + key);
    }
    return value;
}

// Escape caller text, not the template. Omni's reserved delimiters and d1-2B's
// special tokens are structural inputs; allowing caller text to forge one
// changes the answer positions or introduces media without a projector.
std::string escape_omni(std::string text) {
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", i + 1)) {
        size_t j = i + 2;
        while (j < text.size() && ((text[j] >= 'A' && text[j] <= 'Z') || (text[j] >= 'a' && text[j] <= 'z') ||
                                      (text[j] >= '0' && text[j] <= '9') || text[j] == '_'))
            ++j;
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            text.replace(i, j + 2 - i, "<\xC2\xA6" + text.substr(i + 2, j - i - 2) + "\xC2\xA6>");
        }
    }
    string_replace_all(text, "<<d1omni:", "<<d1omni ");
    return text;
}

std::string escape_d1(std::string text, const std::vector<std::string>& specials) {
    bool changed;
    do {
        changed = false;
        std::string out;
        for (size_t i = 0; i < text.size();) {
            const std::string* hit = nullptr;
            for (const auto& special : specials) {
                if (text.compare(i, special.size(), special) == 0) {
                    hit = &special;
                    break;
                }
            }
            if (!hit) {
                out += text[i++];
                continue;
            }
            const unsigned char first = (*hit)[0];
            const size_t        n     = first < 0x80 ? 1 : first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
            out += hit->substr(0, n) + "\xE2\x80\x8B" + hit->substr(n);
            i += hit->size();
            changed = true;
        }
        text = std::move(out);
    } while (changed);
    return text;
}

std::vector<llama_token> single_tokens(const llama_vocab* vocab, const std::vector<std::string>& forms) {
    std::vector<llama_token> result;
    for (const auto& form : forms) {
        const auto tokens = common_tokenize(vocab, form, false, false);
        if (tokens.size() == 1 && std::find(result.begin(), result.end(), tokens[0]) == result.end()) {
            result.push_back(tokens[0]);
        }
    }
    return result;
}

void verbalizers(const llama_vocab* vocab, json& options, DecisionPrepared& output) {
    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; ++c) pool.emplace_back(1, c);
    for (int i = 0; i < 100; ++i) pool.push_back((i < 10 ? "0" : "") + std::to_string(i));
    for (char c = 'a'; c <= 'z'; ++c) pool.emplace_back(1, c);
    for (int i = 0; i < 200; ++i) pool.push_back("#" + std::to_string(i));
    for (char a = 'A'; a <= 'Z'; ++a)
        for (char b = 'A'; b <= 'Z'; ++b) pool.push_back(std::string{a, b});

    const bool               letters = std::all_of(options.begin(), options.end(), [](const json& option) {
        const auto key = option.at("key").get<std::string>();
        return key.size() == 1 && ((key[0] >= 'A' && key[0] <= 'Z') || (key[0] >= 'a' && key[0] <= 'z'));
    });
    std::vector<llama_token> used;
    for (size_t i = 0; i < options.size(); ++i) {
        auto&                    option = options[i];
        const auto               key    = option.at("key").get<std::string>();
        std::vector<llama_token> group;
        std::string              code = key;
        if (output.question_type == "noul") {
            group = single_tokens(vocab,
                key == "true" ? std::vector<std::string>{"yes", "Yes", "YES"}
                              : std::vector<std::string>{"no", "No", "NO"});
        } else if (output.question_type == "score") {
            group = single_tokens(vocab, {key});
        } else {
            if (!letters) {
                if (options.size() <= 26)
                    code = std::string(1, 'A' + i);
                else if (output.type == "lfm3-d1")
                    code = i < 26 ? std::string(1, 'A' + i)
                                  : std::string{
                                        static_cast<char>('A' + (i - 26) / 26), static_cast<char>('A' + (i - 26) % 26)};
                else
                    code = (i < 10 ? "0" : "") + std::to_string(i);
            }
            auto take = [&](const std::string& candidate) {
                const auto tokens = common_tokenize(vocab, candidate, false, false);
                if (tokens.size() != 1 || std::find(used.begin(), used.end(), tokens[0]) != used.end()) return false;
                used.push_back(tokens[0]);
                code  = candidate;
                group = single_tokens(vocab, {candidate, " " + candidate});
                return true;
            };
            bool taken = take(code);
            for (size_t j = 0; !taken && j < pool.size(); ++j) taken = take(pool[j]);
            if (!taken) throw std::invalid_argument("no distinct single-token Liquid label left");
        }
        if (group.empty()) throw std::invalid_argument("Liquid answer has no single-token verbalizer");
        option["label"] = code;
        output.token_groups.push_back(std::move(group));
    }
}

}  // namespace

bool is_liquid_decision(const std::string& type) {
    return type == "lfm2-d1" || type == "lfm2-d1-omni" || type == "lfm3-d1";
}

void configure_decision_context(
    const llama_model* model, llama_context_params& params, bool default_batch, bool default_ubatch) {
    const auto type =
        decision_metadata_string(model, decision_metadata_string(model, "general.architecture") + ".decision.type");
    if (!is_liquid_decision(type)) return;
    if (type == "lfm2-d1-omni" || type == "lfm3-d1") {
        // CLI/server expose context size rather than physical batch size. Make
        // their default fit the context; honor explicit SDK batch limits so a
        // request can fail cleanly before llama.cpp's non-causal batch assert.
        const uint32_t limit = type == "lfm2-d1-omni" ? omni_max_tokens : llama_model_n_ctx_train(model);
        const uint32_t batch = std::min(params.n_ctx, limit);
        if (default_batch) params.n_batch = batch;
        if (default_ubatch) params.n_ubatch = batch;
    }
    if (type == "lfm3-d1") {
        // Each sequence can emit its marked final token, not every input token.
        params.n_outputs_max         = std::max<uint32_t>(1, params.n_seq_max);
        params.n_outputs_max_per_seq = 1;
    }
    if (type == "lfm2-d1-omni") {
        // Unlike vocabulary readout, every option mask needs its own output row.
        params.n_outputs_max         = 0;
        params.n_outputs_max_per_seq = 0;
        params.embeddings            = true;
        params.pooling_type          = LLAMA_POOLING_TYPE_NONE;
    }
}

void prepare_liquid_decision(const llama_model* model, json target, DecisionPrepared& output) {
    const auto*  vocab   = llama_model_get_vocab(model);
    auto&        options = target.at("options");
    const size_t n       = options.size();
    if (n > 255 || (output.question_type == "score" && (n < 2 || n > 10)) ||
        (output.type == "lfm2-d1-omni" && output.question_type == "choice" && n < 2)) {
        throw std::invalid_argument("Liquid supports at most 255 choices and 2 to 10 score levels");
    }
    const bool omni = output.type == "lfm2-d1-omni";
    if (!omni) verbalizers(vocab, options, output);

    std::vector<std::string> specials;
    if (output.type == "lfm3-d1") {
        for (llama_token i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
            if (llama_vocab_get_attr(vocab, i) & LLAMA_TOKEN_ATTR_CONTROL) {
                const std::string text = llama_vocab_get_text(vocab, i);
                if (text.size() > 1 && text.find("\xE2\x80\x8B") == std::string::npos) specials.push_back(text);
            }
        }
        std::stable_sort(
            specials.begin(), specials.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
    }
    if (omni || !specials.empty()) {
        auto escape = [&](const std::string& text) { return omni ? escape_omni(text) : escape_d1(text, specials); };
        for (const auto* key : {"id", "state", "instructions"}) target[key] = escape(target.at(key).get<std::string>());
        for (auto& option : options) {
            option["key"] = escape(option.at("key").get<std::string>());
            if (option.at("description").is_string())
                option["description"] = escape(option.at("description").get<std::string>());
        }
    }
    target["images"] = json::array();
    if (omni) {
        target["audio"]         = false;
        target["sep"]           = omni_sep;
        target["mark_state"]    = omni_state;
        target["mark_question"] = omni_question;
        target["mark_option"]   = omni_option;
    }
    const char* source = llama_model_chat_template(model, "systemone");
    if (!source) throw std::invalid_argument("Liquid model has no systemone template");
    common_chat_template tmpl(source, "", "");
    jinja::context       context(tmpl.source());
    jinja::global_from_json(context, target, false);
    jinja::runtime runtime(context);
    output.prompt = jinja::runtime::gather_string_parts(runtime.execute(tmpl.prog))->as_string().str();

    if (!omni) {
        // The native template includes its own BOS. Adding another changes the
        // forward pass even when the model loads and produces plausible answers.
        output.tokens = common_tokenize(vocab, output.prompt, false, true);
    } else {
        // Segmentation and budgets are part of Omni's training protocol.
        // Joining text before tokenization changes BPE boundaries and can move
        // the mask positions, even when the rendered text looks equivalent.
        const auto pieces = string_split(output.prompt, omni_sep);
        if (pieces.empty() || !pieces[0].empty()) throw std::invalid_argument("invalid Liquid Omni template layout");
        const size_t             budget     = std::max<size_t>(96, std::min<size_t>(24 * n + 32, omni_max_tokens / 2));
        const size_t             per_option = std::max<size_t>(2, (budget - 3 * n) / n);
        std::vector<llama_token> head, state, tail;
        bool                     has_state = false;
        for (size_t i = 1; i < pieces.size(); ++i) {
            std::string piece    = pieces[i];
            size_t      cap      = std::numeric_limits<size_t>::max();
            bool        is_state = false;
            if (string_starts_with(piece, omni_state)) {
                piece.erase(0, omni_state.size());
                is_state = true;
            } else if (string_starts_with(piece, omni_question)) {
                piece.erase(0, omni_question.size());
                cap = std::max<size_t>(16, budget);
            } else if (string_starts_with(piece, omni_option)) {
                piece.erase(0, omni_option.size());
                cap = per_option;
            }
            auto tokens = common_tokenize(vocab, piece, false, true);
            tokens.resize(std::min(tokens.size(), cap));
            if (is_state) {
                if (has_state) throw std::invalid_argument("duplicate Liquid state piece");
                state     = std::move(tokens);
                has_state = true;
            } else {
                auto& dst = has_state ? tail : head;
                dst.insert(dst.end(), tokens.begin(), tokens.end());
            }
        }
        if (!has_state) throw std::invalid_argument("missing Liquid state piece");
        state.resize(std::min(state.size(), omni_max_tokens - std::min(omni_max_tokens, head.size() + tail.size())));
        output.tokens = std::move(head);
        output.tokens.insert(output.tokens.end(), state.begin(), state.end());
        output.tokens.insert(output.tokens.end(), tail.begin(), tail.end());
        output.tokens.resize(std::min(output.tokens.size(), omni_max_tokens));
        const auto mask = llama_vocab_mask(vocab);
        if (mask == LLAMA_TOKEN_NULL) throw std::invalid_argument("Liquid Omni has no mask token");
        for (size_t i = 0; i < output.tokens.size(); ++i)
            if (output.tokens[i] == mask) output.markers.push_back(i);
        if (output.markers.size() != n) throw std::invalid_argument("Liquid options do not fit in the context");
        output.column = output.question_type == "choice" ? 0 : output.question_type == "score" ? 1 : 2;
    }
    const auto        prefix = decision_metadata_string(model, "general.architecture") + ".decision.";
    const std::string bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
    output.temperature       = meta_number(model, prefix + "temperature." + output.question_type, 1.0f);
    output.temperature =
        meta_number(model, prefix + "temperature." + output.question_type + "." + bucket, output.temperature);
    if (!(output.temperature > 0)) throw std::invalid_argument("Liquid temperature must be positive");
    output.score_bias.resize(n, 0.0f);
    if (output.question_type == "score") {
        for (size_t i = 0; i < n; ++i)
            output.score_bias[i] =
                meta_number(model, prefix + "score_bias." + std::to_string(n) + "." + std::to_string(i), 0.0f);
    }
}

std::vector<float> liquid_probabilities(const DecisionPrepared& prepared, const std::vector<float>& scores) {
    if (scores.empty() || scores.size() != prepared.original_index.size() ||
        prepared.score_bias.size() != scores.size() || !std::isfinite(prepared.temperature) ||
        prepared.temperature <= 0) {
        throw std::invalid_argument("invalid Liquid readout shape or temperature");
    }
    std::vector<double> logits(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        logits[i] = static_cast<double>(scores[i]) + prepared.score_bias[i];
        if (std::isnan(logits[i]) || logits[i] == INFINITY) throw std::invalid_argument("invalid Liquid score");
    }
    const double peak = *std::max_element(logits.begin(), logits.end());
    double       sum  = 0;
    for (auto& value : logits) {
        value = std::exp((value - peak) / prepared.temperature);
        sum += value;
    }
    if (!std::isfinite(sum) || sum <= 0) throw std::invalid_argument("invalid Liquid probability sum");
    std::vector<float> probabilities(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
        probabilities.at(prepared.original_index[i]) = static_cast<float>(logits[i] / sum);
    return probabilities;
}

int32_t evaluate_liquid_decision(llama_context* ctx, const DecisionPrepared& prepared, geniex_DecideOutput* output) {
    const bool  omni   = prepared.type == "lfm2-d1-omni";
    const auto& tokens = prepared.tokens;
    if (omni && (llama_model_n_embd_out(llama_get_model(ctx)) != 3 ||
                    llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE || llama_get_causal_attn(ctx))) {
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    if (tokens.empty() || tokens.size() > llama_n_ctx(ctx)) return GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH;
    // Non-causal attention and the memoryless short convolutions must see one
    // isolated question. Splitting it is not a slower equivalent computation.
    const bool whole = !llama_get_causal_attn(ctx) || !llama_get_memory(ctx);
    if (whole && (tokens.size() > llama_n_batch(ctx) || tokens.size() > llama_n_ubatch(ctx))) {
        GENIEX_LOG_WARN(
            "Liquid decision needs {} tokens in one batch; this context allows n_batch={} and "
            "n_ubatch={}. Shorten the question or try --compute cpu (HTTP: compute=cpu); SDK callers "
            "may set explicit batch limits if supported by the device",
            tokens.size(),
            llama_n_batch(ctx),
            llama_n_ubatch(ctx));
        return GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED;
    }
    llama_set_embeddings(ctx, omni);
    common_batch       batch(ctx);
    std::vector<float> scores;
    for (size_t start = 0; start < tokens.size();) {
        const size_t count = std::min<size_t>(llama_n_batch(ctx), tokens.size() - start);
        batch.clear();
        for (size_t i = start; i < start + count; ++i) batch.add(tokens[i], i, 0, omni || i + 1 == tokens.size());
        const int32_t result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (result != 0)
            return result == 1 ? GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG : GENIEX_ERROR_LLM_GENERATION_FAILED;
        start += count;
    }
    if (omni) {
        for (const auto marker : prepared.markers) {
            const float* row = llama_get_embeddings_ith(ctx, static_cast<int32_t>(marker));
            if (!row) return GENIEX_ERROR_COMMON_UNKNOWN;
            scores.push_back(row[prepared.column]);
        }
    } else {
        const float* row = llama_get_logits_ith(ctx, -1);
        if (!row) return GENIEX_ERROR_COMMON_UNKNOWN;
        // These vocabulary forms are trained verbalizers. Unlike generic
        // option codes, they need not all tokenize identically when appended
        // to the prompt; the best form represents the option's score.
        for (const auto& group : prepared.token_groups) {
            float best = -INFINITY;
            for (const auto token : group) best = std::max(best, row[token]);
            scores.push_back(best);
        }
    }
    try {
        const auto probabilities = liquid_probabilities(prepared, scores);
        output->probabilities    = static_cast<float*>(calloc(probabilities.size(), sizeof(float)));
        output->decision_type    = strdup(prepared.type.c_str());
        if (!output->probabilities || !output->decision_type) {
            free(output->probabilities);
            free(output->decision_type);
            *output = {};
            return GENIEX_ERROR_COMMON_MEMORY_ALLOCATION;
        }
        std::copy(probabilities.begin(), probabilities.end(), output->probabilities);
        output->option_count = probabilities.size();
        output->input_tokens = tokens.size();
        output->temperature  = prepared.temperature;
        return GENIEX_SUCCESS;
    } catch (const std::exception& error) {
        GENIEX_LOG_WARN("Liquid decision readout rejected: {}", error.what());
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }
}

}  // namespace geniex
