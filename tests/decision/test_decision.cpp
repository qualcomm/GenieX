// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>

#include "gguf.h"
#include "liquid-decision.h"
#include "logging.h"
#include "mtmd.h"

#ifdef GENIEX_DECISION_REFERENCE_TESTS
#include "server-decision.h"
#endif

// These tests link the adapter directly, without the SDK's model manager.
geniex_log_callback geniex_log       = nullptr;
geniex_LogLevel     geniex_log_level = GENIEX_LOG_LEVEL_ERROR;

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
std::string utf8(int cp) {
    if (cp < 128) return std::string(1, cp);
    return std::string{static_cast<char>(0xc0 | (cp >> 6)), static_cast<char>(0x80 | (cp & 63))};
}

struct FixtureDirectory {
    std::filesystem::path path;
    FixtureDirectory() {
        do {
            path =
                std::filesystem::temp_directory_path() / ("geniex-decision-" + std::to_string(std::random_device{}()));
        } while (!std::filesystem::create_directory(path));
    }
    ~FixtureDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

// A tiny byte-level vocabulary keeps protocol tests model-free. Give it a few
// merged verbalizers so predicates exercise grouping, not just one yes/no form.
llama_model* fixture(const std::string& protocol, const char* source, const char* score_temperature = nullptr) {
    auto* gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "llama");
    gguf_set_val_str(gguf, "llama.decision.type", protocol.c_str());
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(gguf, "tokenizer.ggml.pre", "gpt-2");
    std::vector<std::string> tokens;
    int                      extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool visible = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
        tokens.push_back(utf8(visible ? b : 256 + extra++));
    }
    std::vector<std::string> merges;
    for (const std::string word : {"yes", "Yes", "YES", "no", "No", "NO"}) {
        for (size_t i = 2; i <= word.size(); ++i) {
            tokens.push_back(word.substr(0, i));
            merges.push_back(word.substr(0, i - 1) + " " + word.substr(i - 1, 1));
        }
    }
    const size_t mask = tokens.size();
    tokens.push_back("<|mask|>");
    const size_t bos = tokens.size();
    tokens.push_back("<s>");
    const size_t ordinary = tokens.size();
    tokens.push_back("<ordinary>");
    std::vector<const char*> token_ptrs, merge_ptrs;
    for (const auto& token : tokens) token_ptrs.push_back(token.c_str());
    for (const auto& merge : merges) merge_ptrs.push_back(merge.c_str());
    std::vector<int32_t> types(tokens.size(), 1);
    types[mask] = types[bos] = 3;
    types[ordinary]          = 4;
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
    gguf_set_arr_str(gguf, "tokenizer.ggml.merges", merge_ptrs.data(), merge_ptrs.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    gguf_set_val_u32(gguf, "tokenizer.ggml.mask_token_id", mask);
    gguf_set_val_u32(gguf, "tokenizer.ggml.bos_token_id", bos);
    gguf_set_val_u32(gguf, "tokenizer.ggml.eos_token_id", bos);
    gguf_set_val_bool(gguf, "tokenizer.ggml.add_bos_token", true);
    if (source) gguf_set_val_str(gguf, "tokenizer.chat_template.systemone", source);
    gguf_set_val_f32(gguf, "llama.decision.temperature.score", 3.0f);
    gguf_set_val_f32(gguf, "llama.decision.temperature.score.2", 2.0f);
    if (score_temperature) gguf_set_val_str(gguf, "llama.decision.temperature.score.2", score_temperature);
    gguf_set_val_f32(gguf, "llama.decision.score_bias.2.1", 1.0f);
    const FixtureDirectory directory;
    const auto             path = (directory.path / "fixture.gguf").string();
    check(gguf_write_to_file(gguf, path.c_str(), true), "write fixture");
    gguf_free(gguf);
    auto params       = llama_model_default_params();
    params.vocab_only = true;
    auto* model       = llama_model_load_from_file(path.c_str(), params);
    check(model != nullptr, "load fixture");
    return model;
}

const char* options_template =
    "{{ state }}|{{ type }}|{% for o in options %}{{ o.key }}={{ o.label }}:{{ o.description }};{% endfor %}"
    "{% for q in questions %}question={{ q.id }};{% endfor %}";
const char* omni_template =
    "{{ sep }}<s>{{ sep }}{{ mark_state }}{{ state }}{{ sep }}{{ mark_question }}{{ instructions }}{% for o in options "
    "%}{{ sep }}<|mask|>{{ sep }}{{ mark_option }} {{ o.key }}:{{ o.description }}{% endfor %}";

void protocols() {
    geniex_DecideOption   options[] = {{"false", false, nullptr}, {"true", false, nullptr}};
    geniex_DecideQuestion question{"q", "predicate", "Does it hold?", options, 2};
    geniex_DecideInput    input{};
    input.state          = "state";
    input.questions      = &question;
    input.question_count = 1;
    for (const std::string type : {"openjev", "lev", "nimble", "", "lfm2-d1", "lfm3-d1", "lfm2-d1-omni", "unknown"}) {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            fixture(type, type == "lfm2-d1-omni" ? omni_template : options_template), llama_model_free);
        for (uint32_t sequences : {1u, 2u, 4u}) {
            auto context_params                  = llama_context_default_params();
            context_params.n_seq_max             = sequences;
            context_params.n_outputs_max         = sequences;
            context_params.n_outputs_max_per_seq = 1;
            geniex::configure_decision_context(model.get(), context_params, false, false);
            if (type == "lfm2-d1-omni") {
                check(context_params.n_outputs_max == 0 && context_params.n_outputs_max_per_seq == 0 &&
                          context_params.embeddings && context_params.pooling_type == LLAMA_POOLING_TYPE_NONE,
                    "Omni reserves all mask rows");
            } else {
                check(context_params.n_outputs_max >= context_params.n_seq_max &&
                          context_params.n_outputs_max_per_seq == 1,
                    "logit protocols reserve one output row per sequence");
            }
        }
        geniex::DecisionPrepared prepared;
        const auto               rc = geniex::prepare_decision(model.get(), &input, &prepared);
        if (type == "unknown") {
            check(rc == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED, "unknown rejected");
            continue;
        }
        check(rc == GENIEX_SUCCESS, "prepare protocol");
        check(prepared.type == (type.empty() ? "openjev" : type), "detected protocol");
        const std::string literal_media = "literal " + std::string(mtmd_default_marker()) + " text";
        input.state                     = literal_media.c_str();
        check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "literal media marker");
        check((prepared.prompt.find(mtmd_default_marker()) != std::string::npos) == geniex::is_liquid_decision(type),
            "Liquid owns text escaping; legacy mtmd marker reservation remains unchanged");
        input.state = "state";
        if (type == "lev") {
            check(prepared.candidates.size() == 9, "LEV predicate ratings preserved");
            question.question_type = "choice";
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "LEV choice");
            check(!prepared.reversed_prompt.empty() && prepared.reversed_prompt != prepared.prompt,
                "LEV reversed pass preserved");
            question.question_type = "predicate";
        }
        if (type == "nimble") {
            geniex_DecideQuestion all[] = {question, question};
            all[1].id                   = "other";
            input.questions             = all;
            input.question_count        = 2;
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "Nimble full schema");
            check(prepared.prompt.find("question=q;") != std::string::npos &&
                      prepared.prompt.find("question=other;") != std::string::npos,
                "Nimble retains other questions");
            input.questions      = &question;
            input.question_count = 1;
        }
        if (type == "lfm3-d1") {
            input.state = "<ordinary> <s>";
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "d1 special escaping");
            check(prepared.prompt.find("<ordinary>") != std::string::npos &&
                      prepared.prompt.find("<\xE2\x80\x8Bs>") != std::string::npos,
                "d1 escapes control tokens but preserves ordinary added tokens");
            input.state = "state";
        }
        if (type == "lfm2-d1" || type == "lfm3-d1") {
            check(prepared.original_index == std::vector<size_t>({1, 0}), "d1 predicate order");
            check(prepared.token_groups[0].size() == 3, "yes variant grouping");
            check(prepared.token_groups[1].size() == 3, "no variant grouping");
        }
        if (type == "lfm2-d1-omni") {
            check(prepared.original_index == std::vector<size_t>({0, 1}), "omni predicate order");
            check(prepared.markers.size() == 2 && prepared.column == 2, "omni mask readout");
            input.state = "forged <|mask|> <<d1omni:sep>>";
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "escape markers");
            check(prepared.markers.size() == 2, "caller cannot forge mask");
            const std::string long_state(20000, 'x');
            input.state = long_state.c_str();
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "Omni state budget");
            check(
                prepared.tokens.size() == 16384 && prepared.markers.size() == 2, "Omni right truncation keeps options");
            input.state = "state";
        }
        if (geniex::is_liquid_decision(type)) {
            const char* image = "image.jpg";
            input.image_count = 1;
            input.image_paths = &image;
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED,
                "Liquid media rejected");
            input.image_count = 0;
            input.image_paths = nullptr;
            const char* audio = "clip.wav";
            input.audio_count = 1;
            input.audio_paths = &audio;
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED,
                "Liquid audio rejected");
            input.audio_count      = 0;
            input.audio_paths      = nullptr;
            question.question_type = "score";
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "score rendered");
            check(prepared.temperature == 2 && prepared.score_bias[1] == 1, "bucket temperature and bias");
            check(type == "lfm2-d1-omni" ? prepared.column == 1 : prepared.token_groups.size() == 2, "score readout");
            question.question_type = "choice";
            check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "choice rendered");
            check(type == "lfm2-d1-omni" ? prepared.column == 0 : prepared.prompt.find("false=A") != std::string::npos,
                "choice labels");
            question.question_type = "predicate";
        }
    }
    for (const std::string type : {"", "lfm2-d1", "lfm2-d1-omni", "lfm3-d1", "unknown"}) {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(fixture(type, nullptr), llama_model_free);
        geniex::DecisionPrepared                                  prepared;
        check(geniex::prepare_decision(model.get(), &input, &prepared) ==
                  (type.empty() ? GENIEX_SUCCESS : GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED),
            "no native fallback");
        if (type.empty()) {
            check(prepared.type == "generic", "generic preserved");
            check(prepared.prompt.find("A. false") != std::string::npos, "generic labeled prompt");
        }
    }
}

void calibration() {
    geniex_DecideOption   options[] = {{"low", false, nullptr}, {"high", false, nullptr}};
    geniex_DecideQuestion question{"q", "score", "How urgent?", options, 2};
    geniex_DecideInput    input{};
    input.state                 = "state";
    input.questions             = &question;
    input.question_count        = 1;
    const std::string truncated = "1." + std::string(300, '0') + "garbage";
    for (const char* invalid : {"", "nan", "0", "-1", "1trailing", truncated.c_str()}) {
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            fixture("lfm2-d1", options_template, invalid), llama_model_free);
        geniex::DecisionPrepared prepared;
        check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_ERROR_COMMON_INVALID_INPUT,
            "malformed calibration metadata must not use a default");
    }
    geniex::DecisionPrepared prepared;
    prepared.original_index  = {1, 0};
    prepared.score_bias      = {0, 1};
    prepared.temperature     = 2;
    const auto probabilities = geniex::liquid_probabilities(prepared, {3, 0});
    check(std::abs(probabilities[1] - 0.73105858f) < 1e-6, "bias before temperature and caller mapping");
    check(std::abs(probabilities[0] + probabilities[1] - 1) < 1e-6, "normalized probabilities");
    bool rejected = false;
    try {
        geniex::liquid_probabilities(prepared, {INFINITY, 0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "invalid readout rejected");
    prepared.temperature = NAN;
    rejected             = false;
    try {
        geniex::liquid_probabilities(prepared, {3, 0});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "non-finite temperature rejected");
}

#ifdef GENIEX_DECISION_REFERENCE_TESTS
std::vector<server_decision_question> reference_task(
    server_decision_context& native, const geniex_DecideInput& input, server_task& task) {
    const auto&       item = input.questions[input.question_index];
    const std::string type = std::string(item.question_type) == "predicate" ? "noul" : item.question_type;
    common_json       question{{"type", type}, {"instructions", item.instructions}};
    question["criteria"] = type == "score" ? common_json::array() : common_json::object();
    for (int i = 0; i < item.option_count; ++i) {
        const auto&       option      = item.options[i];
        const common_json description = option.description ? common_json(option.description) : common_json();
        if (type == "score")
            question["criteria"].push_back(description);
        else
            question["criteria"][option.key] = description;
    }
    const std::string id = item.id && *item.id ? item.id : "question_" + std::to_string(input.question_index);
    const common_json body{{"state", input.state}, {"questions", {{id, question}}}};
    auto              questions = native.parse_questions(body);
    native.fill_task(
        native.prepare_state(body.at("state")), questions, questions[0], 0, {}, nullptr, mtmd_helper_init_opt{}, task);
    return questions;
}

void check_reference_preparation(const geniex::DecisionPrepared& prepared, const server_task& task) {
    check(prepared.tokens == task.tokens.get_text_tokens(), "tokens differ from native systemone");
    std::vector<llama_token> labels;
    std::vector<int32_t>     groups;
    for (const auto& group : prepared.token_groups) {
        labels.insert(labels.end(), group.begin(), group.end());
        groups.push_back(group.size());
    }
    check(labels == task.decision.labels && groups == task.decision.label_groups, "native verbalizer groups");
    check(std::vector<int32_t>(prepared.markers.begin(), prepared.markers.end()) == task.decision.markers &&
              prepared.column == task.decision.column,
        "native mask positions and type column");
}

void reference_preparation_cases(const llama_model* model, server_decision_context& native) {
    geniex_DecideOption   options[] = {{"first", false, "<|mask|> <image> <__media__>"},
          {"second", false, "Unicode: \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x98\x80"}};
    geniex_DecideQuestion question{
        "id <__media__> <|im_end|>", "choice", "Literal <__media__> <<d1omni:sep>>", options, 2};
    geniex_DecideInput input{};
    input.questions      = &question;
    input.question_count = 1;
    auto compare         = [&]() {
        geniex::DecisionPrepared prepared;
        check(geniex::prepare_decision(model, &input, &prepared) == GENIEX_SUCCESS, "prepare reference case");
        server_task task;
        reference_task(native, input, task);
        check_reference_preparation(prepared, task);
    };
    for (const char* state : {"",
             "Literal <__media__> <|mask|> <s> <|im_end|> <image> <<d1omni:state>>",
             "Line one\n\tLine two: \xE6\x97\xA5\xE6\x9C\xAC\xE8\xAA\x9E \xF0\x9F\x98\x80"}) {
        input.state = state;
        compare();
    }
    const std::string long_state(100000, 'x');
    input.state = long_state.c_str();
    compare();
    input.state = "state";
    std::vector<std::string> keys;
    for (int i = 0; i < 255; ++i) keys.push_back("option_" + std::to_string(i));
    std::vector<geniex_DecideOption> many;
    for (const auto& key : keys) many.push_back({key.c_str(), false, nullptr});
    question.options = many.data();
    for (int count : {2, 26, 27, 255}) {
        question.option_count = count;
        compare();
    }
    question.question_type = "score";
    for (int count : {2, 3, 5, 6, 10}) {
        question.option_count = count;
        compare();
    }
}
#endif

void inference(const char* path) {
    auto params         = llama_model_default_params();
    params.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(path, params), llama_model_free);
    check(model != nullptr, "load inference model");
    auto cparams  = llama_context_default_params();
    cparams.n_ctx = cparams.n_batch = cparams.n_ubatch = 4096;
    cparams.n_threads = cparams.n_threads_batch = 4;
    geniex::configure_decision_context(model.get(), cparams);
    const auto protocol = geniex::decision_metadata_string(
        model.get(), geniex::decision_metadata_string(model.get(), "general.architecture") + ".decision.type");
    if (protocol == "lfm3-d1")
        check(cparams.n_outputs_max == 1 && cparams.n_outputs_max_per_seq == 1, "d1-2B reserves one logits row");
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cparams), llama_free);
    check(ctx != nullptr, "create inference context");
    geniex_DecideOption predicate[] = {{"false", false, nullptr}, {"true", false, nullptr}};
    geniex_DecideOption choices[]   = {{"billing", false, "Charges, refunds, invoices"},
          {"technical", false, "App or site faults"},
          {"fraud", false, "Suspected unauthorised use"}};
    geniex_DecideOption levels[]    = {
        {"low", false, "Can wait"}, {"medium", false, "Today"}, {"high", false, "Blocking the customer now"}};
    geniex_DecideQuestion questions[] = {{"refund", "predicate", "Is the customer asking for a refund?", predicate, 2},
        {"team", "choice", "Which team should handle this?", choices, 3},
        {"urgency", "score", "How urgent is this?", levels, 3}};
    geniex_DecideInput    input{};
    const std::string     state = "I was charged twice this month, please refund one of them.";
    std::string           long_state;
    for (int i = 0; i < 100; ++i) long_state += "Background context. ";
    long_state += state;
    input.state          = state.c_str();
    input.questions      = questions;
    input.question_count = 3;
    common_json results  = common_json::array();
#ifdef GENIEX_DECISION_REFERENCE_TESTS
    server_decision_context native;
    native.init(model.get());
    reference_preparation_cases(model.get(), native);
#endif
    for (int i = 0; i < 5; ++i) {
        input.question_index = i < 3 ? i : 0;
        input.state          = i == 4 ? long_state.c_str() : state.c_str();
        if (i == 4 && results[0]["type"] == "lfm2-d1") {
            // Causal d1 can split prefill; exercise a logical and physical batch
            // boundary rather than only the one-batch non-causal models.
            ctx.reset();
            cparams.n_batch  = 128;
            cparams.n_ubatch = 64;
            ctx.reset(llama_init_from_model(model.get(), cparams));
            check(ctx != nullptr, "causal chunked context");
        }
        geniex::DecisionPrepared prepared;
        check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "prepare inference");
        if (i == 4) check(prepared.tokens.size() > 128, "long inference crosses batch boundary");
#ifdef GENIEX_DECISION_REFERENCE_TESTS
        server_task task;
        const auto  native_questions = reference_task(native, input, task);
        check_reference_preparation(prepared, task);
#endif
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        geniex_DecideOutput output{};
        check(geniex::evaluate_liquid_decision(ctx.get(), prepared, &output) == GENIEX_SUCCESS, "native inference");
#ifdef GENIEX_DECISION_REFERENCE_TESTS
        std::vector<float> native_scores;
        if (!task.decision.labels.empty()) {
            const float* row = llama_get_logits_ith(ctx.get(), -1);
            size_t       j   = 0;
            for (const int count : task.decision.label_groups) {
                float best = -INFINITY;
                for (int k = 0; k < count; ++k) best = std::max(best, row[task.decision.labels[j++]]);
                native_scores.push_back(best);
            }
        } else {
            for (const auto marker : task.decision.markers) {
                native_scores.push_back(llama_get_embeddings_ith(ctx.get(), marker)[task.decision.column]);
            }
        }
        const auto answer = native.format_answer(native_questions[0], {native_scores});
        for (size_t j = 0; j < prepared.original_index.size(); ++j) {
            const double probability =
                input.question_index == 0
                    ? (native_questions[0].options[j].key == "true" ? answer.at("noul").get<double>()
                                                                    : 1 - answer.at("noul").get<double>())
                    : answer.at("probabilities").at(native_questions[0].options[j].key).get<double>();
            check(std::abs(probability - output.probabilities[prepared.original_index[j]]) < 1e-6,
                "calibration differs from systemone");
        }
#endif
        const std::vector<float> probabilities(output.probabilities, output.probabilities + output.option_count);
        check(std::all_of(probabilities.begin(),
                  probabilities.end(),
                  [](float p) { return std::isfinite(p) && p >= 0 && p <= 1; }),
            "invalid inference probabilities");
        if (i == 3) {
            const auto first = results[0]["probabilities"].get<std::vector<float>>();
            for (size_t j = 0; j < first.size(); ++j)
                check(std::abs(first[j] - probabilities[j]) < 1e-6, "repeated question must not inherit prior state");
        }
        results.push_back(common_json{{"type", output.decision_type},
            {"case", std::vector<std::string>{"predicate", "choice", "score", "repeat", "long"}[i]},
            {"question", questions[input.question_index].id},
            {"input_tokens", output.input_tokens},
            {"temperature", output.temperature},
            {"probabilities", probabilities}});
        free(output.probabilities);
        free(output.decision_type);
    }
    if (protocol == "lfm3-d1") {
        for (uint32_t sequences : {2u, 4u}) {
            ctx.reset();
            cparams.n_seq_max = sequences;
            geniex::configure_decision_context(model.get(), cparams);
            check(cparams.n_outputs_max >= sequences && cparams.n_outputs_max_per_seq == 1,
                "multi-sequence context reserves one output row per sequence");
            ctx.reset(llama_init_from_model(model.get(), cparams));
            check(ctx != nullptr, "multi-sequence d1-2B context loads");
            input.state = state.c_str();
            for (int question = 0; question < 3; ++question) {
                input.question_index = question;
                geniex::DecisionPrepared prepared;
                check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS,
                    "multi-sequence preparation");
                geniex_DecideOutput output{};
                check(geniex::evaluate_liquid_decision(ctx.get(), prepared, &output) == GENIEX_SUCCESS,
                    "multi-sequence readout");
                const auto               first = results[question]["probabilities"].get<std::vector<float>>();
                const std::vector<float> probabilities(
                    output.probabilities, output.probabilities + output.option_count);
                check(first == probabilities, "sequence capacity must not change probabilities");
                free(output.probabilities);
                free(output.decision_type);
            }
        }
        ctx.reset();
        cparams.n_seq_max = 1;
        cparams.n_ctx     = 16384;
        geniex::configure_decision_context(model.get(), cparams);
        check(cparams.n_outputs_max == 1 && cparams.n_outputs_max_per_seq == 1,
            "large d1-2B context still reserves one logits row");
        ctx.reset(llama_init_from_model(model.get(), cparams));
        check(ctx != nullptr, "large single-row context");
        input.state          = state.c_str();
        input.question_index = 0;
        geniex::DecisionPrepared prepared;
        check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "large-context preparation");
        geniex_DecideOutput output{};
        check(geniex::evaluate_liquid_decision(ctx.get(), prepared, &output) == GENIEX_SUCCESS,
            "large-context single-row readout");
        const auto first = results[0]["probabilities"].get<std::vector<float>>();
        for (size_t i = 0; i < first.size(); ++i)
            check(std::abs(first[i] - output.probabilities[i]) < 1e-6, "single-row context-size parity");
        free(output.probabilities);
        free(output.decision_type);
        ctx.reset();
        cparams.n_ctx = 4096;
        geniex::configure_decision_context(model.get(), cparams);
        ctx.reset(llama_init_from_model(model.get(), cparams));
        check(ctx != nullptr, "restore context-overflow test limit");
    }
    std::string oversized;
    for (int i = 0; i < 10000; ++i) oversized += " background.";
    input.state = oversized.c_str();
    geniex::DecisionPrepared too_long;
    check(geniex::prepare_decision(model.get(), &input, &too_long) == GENIEX_SUCCESS, "prepare context overflow");
    geniex_DecideOutput rejected{};
    check(geniex::evaluate_liquid_decision(ctx.get(), too_long, &rejected) ==
              GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH,
        "context overflow rejected before forward");
    input.state = state.c_str();
    if (results[0]["type"] != "lfm2-d1") {
        auto safety_defaults     = cparams;
        safety_defaults.n_ubatch = 128;
        geniex::configure_decision_context(model.get(), safety_defaults, true, false);
        check(safety_defaults.n_ubatch == 128, "accelerator microbatch safety default retained");
        // A deliberately small physical batch must fail before the runtime's
        // non-causal assertion, not silently split the native forward pass.
        ctx.reset();
        cparams.n_batch = cparams.n_ubatch = 8;
        geniex::configure_decision_context(model.get(), cparams, false, false);
        ctx.reset(llama_init_from_model(model.get(), cparams));
        check(ctx != nullptr, "small-batch context");
        input.question_index = 0;
        geniex::DecisionPrepared prepared;
        check(geniex::prepare_decision(model.get(), &input, &prepared) == GENIEX_SUCCESS, "prepare small-batch test");
        geniex_DecideOutput output{};
        check(geniex::evaluate_liquid_decision(ctx.get(), prepared, &output) == GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED,
            "non-causal batch constraint");
    }
    std::printf("%s\n", results.dump().c_str());
}
}  // namespace

int main(int argc, char** argv) {
    llama_backend_init();
    try {
        if (argc == 2)
            inference(argv[1]);
        else {
            protocols();
            calibration();
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
