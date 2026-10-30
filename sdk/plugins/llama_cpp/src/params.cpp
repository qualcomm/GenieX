// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#include "params.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "common.h"
#include "geniex_model.h"
#include "ggml-backend.h"
#include "logging.h"
#include "speculative.h"

namespace geniex {

// Platform

enum class Platform { Linux, Windows, Android };

constexpr Platform kHostPlatform =
#if defined(__ANDROID__)
    Platform::Android;
#elif defined(_WIN32)
    Platform::Windows;
#else
    Platform::Linux;
#endif

// Device

Device classify_device(const char* device_id, int n_gpu_layers) {
    if (n_gpu_layers == 0) return Device::CPU;                   // <0 means "all layers" to llama.cpp
    if (!device_id || device_id[0] == '\0') return Device::NPU;  // HYBRID treat as NPU
    const std::string id(device_id);
    if (id.rfind("GPU", 0) == 0) return Device::GPU;
    if (id.rfind("HTP", 0) == 0) return Device::NPU;
    return Device::CPU;  // UNKOWN treat as CPU
}

int resolve_n_threads(int requested) { return requested > 0 ? requested : common_cpu_get_num_math(); }

llama_model_params build_model_params(const geniex_ModelConfig& config, Device device) {
    static const bool mmap_matrix[3][3] = {
        {true, false, false},   // Linux
        {false, false, false},  // Windows
        {false, false, false}   // Android
    };

    bool use_mmap = mmap_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    llama_model_params mpar = llama_model_default_params();
    mpar.load_mode          = use_mmap ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    mpar.n_gpu_layers       = config.n_gpu_layers;
    GENIEX_LOG_INFO("[Optimise] model params: n_gpu_layers={}, load_mode={}",
        mpar.n_gpu_layers,
        llama_load_mode_name(mpar.load_mode));
    return mpar;
}

std::optional<common_params_speculative> build_speculative_params(const geniex_ModelConfig& config) {
    if (!config.spec_type || config.spec_type[0] == '\0' || strcmp(config.spec_type, "none") == 0) {
        return std::nullopt;
    }

    std::vector<common_speculative_type> types =
        common_speculative_types_from_names(string_split<std::string>(config.spec_type, ','));
    types.erase(std::remove(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_NONE), types.end());
    if (types.empty()) {
        GENIEX_LOG_ERROR("unrecognized --spec-type: '{}'", config.spec_type);
        return std::nullopt;
    }

    common_params_speculative spar;
    spar.types       = types;
    spar.draft.n_max = config.spec_n_max > 0 ? config.spec_n_max : 3;
    if (config.spec_n_min > 0) spar.draft.n_min = config.spec_n_min;
    if (config.spec_p_min > 0) spar.draft.p_min = config.spec_p_min;
    return spar;
}

llama_context_params build_context_params(const geniex_ModelConfig& config, int32_t n_ctx_default, Device device,
    const common_params_speculative* spec, OutputReservation output_reservation) {
    // Linux NPU (IQ9/IQ8) is pinned to 128: ggml-hexagon's per-ubatch compute buffer can fail to
    // fastrpc_mmap as one contiguous block at larger sizes on these HTP targets, independent of
    // n_batch/n_ctx. See qcom-ai-hub/geniex#1683.
    static const uint32_t ubatch_matrix[3][3] = {
        {2048, 512, 128},   // Linux
        {2048, 512, 1024},  // Windows
        {1024, 512, 1024}   // Android
    };
    static const bool fa_matrix[3][3] = {
        {true, false, true},  // Linux
        {true, false, true},  // Windows
        {true, false, true}   // Android
    };

    uint32_t ubatch = ubatch_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];
    bool     fa     = fa_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    llama_context_params cpar = llama_context_default_params();
    cpar.swa_full             = false;
    cpar.n_ctx                = config.n_ctx > 0 ? config.n_ctx : n_ctx_default;
    cpar.n_batch              = config.n_batch > 0 ? config.n_batch : 2048;
    cpar.n_ubatch             = config.n_ubatch > 0 ? config.n_ubatch : ubatch;
    cpar.n_seq_max            = config.n_seq_max > 0 ? config.n_seq_max : 1;
    cpar.n_threads            = resolve_n_threads(config.n_threads);
    cpar.n_threads_batch      = resolve_n_threads(config.n_threads_batch);
    cpar.flash_attn_type      = static_cast<llama_flash_attn_type>(fa);
    cpar.no_perf              = false;

    if (spec) {
        cpar.n_rs_seq     = spec->need_n_rs_seq();
        const auto limits = common_speculative_get_output_limits(
            static_cast<int32_t>(cpar.n_batch), static_cast<int32_t>(cpar.n_seq_max), common_speculative_n_max(spec));
        cpar.n_outputs_max         = std::max<int32_t>(1, limits.total);
        cpar.n_outputs_max_per_seq = std::max<int32_t>(1, limits.per_seq);
    } else if (output_reservation == OutputReservation::LastPerSequence) {
        // Reserve one row per sequence instead of one row per ubatch token.
        cpar.n_outputs_max         = std::max<uint32_t>(1, cpar.n_seq_max);
        cpar.n_outputs_max_per_seq = 1;
    }

    GENIEX_LOG_INFO(
        "[Optimise] context params: n_ctx={}, n_batch={}, n_ubatch={}, n_seq_max={}, n_threads={}, "
        "n_threads_batch={}, flash_attn_type={}, swa_full={}, kv_unified={}, no_perf={}, n_rs_seq={}, "
        "n_outputs_max={}, n_outputs_max_per_seq={}",
        cpar.n_ctx,
        cpar.n_batch,
        cpar.n_ubatch,
        cpar.n_seq_max,
        cpar.n_threads,
        cpar.n_threads_batch,
        static_cast<int>(cpar.flash_attn_type),
        cpar.swa_full,
        cpar.kv_unified,
        cpar.no_perf,
        cpar.n_rs_seq,
        cpar.n_outputs_max,
        cpar.n_outputs_max_per_seq);
    return cpar;
}

void warmup_context(const llama_model* model, llama_context* ctx) {
    if (llama_model_has_encoder(model) || !llama_model_has_decoder(model)) return;

    const llama_vocab*       vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens;
    if (llama_vocab_bos(vocab) != LLAMA_TOKEN_NULL) tokens.push_back(llama_vocab_bos(vocab));
    if (llama_vocab_eos(vocab) != LLAMA_TOKEN_NULL) tokens.push_back(llama_vocab_eos(vocab));
    if (tokens.empty()) tokens.push_back(0);

    // llama_decode asserts n_tokens <= n_batch, so clamp like common_init_from_params does.
    const auto n_tokens = std::min<size_t>(tokens.size(), llama_n_batch(ctx));
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), static_cast<int32_t>(n_tokens))) != 0) {
        GENIEX_LOG_WARN("model warmup decode failed");
    }
    llama_synchronize(ctx);
    llama_memory_clear(llama_get_memory(ctx), /*clear data=*/true);
    llama_perf_context_reset(ctx);
}

ggml_threadpool_params build_threadpool_params(int n_threads, Device device) {
    // Linux / NPU stays unpinned: strict-pinning 6 threads to cores 2-7 halves
    // HTP decode throughput on Snapdragon X2 Elite (10.6 vs 20.1 tok/s), and
    // llama.cpp's own Snapdragon Linux runs never pin.
    static const bool pin_matrix[3][3] = {
        {false, true, false},   // Linux
        {false, false, false},  // Windows
        {true, true, true}      // Android
    };
    static const uint32_t poll_matrix[3][3] = {
        {0, 1000, 1000},    // Linux
        {0, 0, 0},          // Windows
        {1000, 1000, 1000}  // Android
    };

    uint32_t poll = poll_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];
    bool     pin  = pin_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);

    if (pin) {
        // Reserve up to 2 cores only if n_threads leaves that slack, to avoid pinning past
        // the real core count or oversubscribing it now that n_threads can reach hardware_concurrency().
        int hw_threads     = static_cast<int>(std::thread::hardware_concurrency());
        int reserved_cores = hw_threads > 0 ? std::max(0, std::min(2, hw_threads - n_threads)) : 2;
        for (int i = 0; i < n_threads; ++i) {
            tpp.cpumask[reserved_cores + i] = true;
        }
        tpp.strict_cpu = true;
    }

    if (poll > 0) {
        tpp.poll = poll;
    }

    return tpp;
}

common_params_sampling build_sampling_params(const geniex_SamplerConfig* cfg) {
    common_params_sampling s;
    if (!cfg) return s;  // null cfg -> upstream defaults

    s.seed            = (cfg->seed != 0) ? cfg->seed : LLAMA_DEFAULT_SEED;
    s.top_k           = (cfg->top_k != 0) ? cfg->top_k : 40;
    s.top_p           = (cfg->top_p != 0.0f) ? cfg->top_p : 0.95f;
    s.min_p           = (cfg->min_p != 0.0f) ? cfg->min_p : 0.05f;
    s.temp            = (cfg->temperature != 0.0f) ? cfg->temperature : 0.8f;
    s.penalty_repeat  = (cfg->repetition_penalty != 0.0f) ? cfg->repetition_penalty : 1.0f;
    s.penalty_present = (cfg->presence_penalty != 0.0f) ? cfg->presence_penalty : 0.0f;
    s.penalty_freq    = (cfg->frequency_penalty != 0.0f) ? cfg->frequency_penalty : 0.0f;

    // Grammar: prefer an inline string, otherwise read it from the grammar file.
    if (cfg->grammar_string && strlen(cfg->grammar_string) > 0) {
        s.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, cfg->grammar_string);
        GENIEX_LOG_DEBUG("Applied grammar string: {}", cfg->grammar_string);
    } else if (cfg->grammar_path && strlen(cfg->grammar_path) > 0) {
        std::ifstream file(cfg->grammar_path);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            s.grammar = common_grammar(COMMON_GRAMMAR_TYPE_USER, buffer.str());
            GENIEX_LOG_DEBUG("Applied grammar from file: {}", cfg->grammar_path);
        } else {
            GENIEX_LOG_ERROR("Failed to read grammar file: {}", cfg->grammar_path);
        }
    }

    const auto& chat_grammar = cfg->chat_template_grammar;
    if (chat_grammar.grammar_type != GENIEX_GRAMMAR_TYPE_NONE) {
        if (!s.grammar.empty()) {
            throw std::invalid_argument("Cannot specify grammar with tools");
        }
        if (chat_grammar.grammar_type != GENIEX_GRAMMAR_TYPE_TOOL_CALLS || !chat_grammar.grammar ||
            chat_grammar.grammar[0] == '\0') {
            throw std::invalid_argument("invalid chat-template grammar");
        }
        if (chat_grammar.trigger_count < 0 || chat_grammar.preserved_token_count < 0 ||
            (chat_grammar.trigger_count > 0 && !chat_grammar.triggers) ||
            (chat_grammar.preserved_token_count > 0 && !chat_grammar.preserved_tokens)) {
            throw std::invalid_argument("invalid chat-template grammar metadata");
        }

        s.grammar      = {COMMON_GRAMMAR_TYPE_TOOL_CALLS, chat_grammar.grammar};
        s.grammar_lazy = chat_grammar.lazy;
        if (chat_grammar.generation_prompt) {
            s.generation_prompt = chat_grammar.generation_prompt;
        }
        for (int32_t i = 0; i < chat_grammar.trigger_count; ++i) {
            const auto&            source = chat_grammar.triggers[i];
            common_grammar_trigger trigger;
            switch (source.trigger_type) {
                case GENIEX_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    break;
                case GENIEX_GRAMMAR_TRIGGER_TYPE_WORD:
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_WORD;
                    break;
                case GENIEX_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN;
                    break;
                case GENIEX_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                    trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
                    break;
                default:
                    throw std::invalid_argument("invalid chat-template grammar trigger type");
            }
            if (source.value) {
                trigger.value = source.value;
            }
            trigger.token = source.token;
            s.grammar_triggers.push_back(std::move(trigger));
        }
        for (int32_t i = 0; i < chat_grammar.preserved_token_count; ++i) {
            s.preserved_tokens.insert(chat_grammar.preserved_tokens[i]);
        }
    }

    return s;
}

void free_chat_template_grammar(geniex_ChatTemplateGrammar* grammar) {
    if (!grammar) {
        return;
    }
    free(grammar->grammar);
    free(grammar->generation_prompt);
    for (int32_t i = 0; i < grammar->trigger_count; ++i) {
        free(grammar->triggers[i].value);
    }
    free(grammar->triggers);
    free(grammar->preserved_tokens);
    *grammar = {};
}

bool copy_chat_template_grammar(const common_params_sampling& sampling, geniex_ChatTemplateGrammar* output) {
    if (!output) {
        return false;
    }
    *output = {};
    if (sampling.grammar.type != COMMON_GRAMMAR_TYPE_TOOL_CALLS || sampling.grammar.empty()) {
        return true;
    }

    output->grammar_type      = GENIEX_GRAMMAR_TYPE_TOOL_CALLS;
    output->grammar           = strdup(common_grammar_value(sampling.grammar).c_str());
    output->lazy              = sampling.grammar_lazy;
    output->generation_prompt = strdup(sampling.generation_prompt.c_str());
    if (!output->grammar || !output->generation_prompt) {
        free_chat_template_grammar(output);
        return false;
    }

    if (!sampling.grammar_triggers.empty()) {
        output->triggers = static_cast<geniex_GrammarTrigger*>(
            calloc(sampling.grammar_triggers.size(), sizeof(geniex_GrammarTrigger)));
        if (!output->triggers) {
            free_chat_template_grammar(output);
            return false;
        }
        output->trigger_count = static_cast<int32_t>(sampling.grammar_triggers.size());
        for (int32_t i = 0; i < output->trigger_count; ++i) {
            const auto& source = sampling.grammar_triggers[i];
            auto&       target = output->triggers[i];
            switch (source.type) {
                case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                    target.trigger_type = GENIEX_GRAMMAR_TRIGGER_TYPE_TOKEN;
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                    target.trigger_type = GENIEX_GRAMMAR_TRIGGER_TYPE_WORD;
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                    target.trigger_type = GENIEX_GRAMMAR_TRIGGER_TYPE_PATTERN;
                    break;
                case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                    target.trigger_type = GENIEX_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL;
                    break;
                default:
                    free_chat_template_grammar(output);
                    return false;
            }
            target.value = strdup(source.value.c_str());
            target.token = source.token;
            if (!target.value) {
                free_chat_template_grammar(output);
                return false;
            }
        }
    }

    if (!sampling.preserved_tokens.empty()) {
        output->preserved_tokens = static_cast<int32_t*>(calloc(sampling.preserved_tokens.size(), sizeof(int32_t)));
        if (!output->preserved_tokens) {
            free_chat_template_grammar(output);
            return false;
        }
        output->preserved_token_count = static_cast<int32_t>(sampling.preserved_tokens.size());
        int32_t i                     = 0;
        for (const llama_token token : sampling.preserved_tokens) {
            output->preserved_tokens[i++] = token;
        }
    }

    return true;
}

std::optional<std::vector<ggml_backend_dev_t>> resolve_devices(const char* device_id) {
    std::vector<ggml_backend_dev_t> devices;
    if (!device_id || device_id[0] == '\0') {
        return devices;  // empty: caller uses the model default
    }

    const std::string device_str(device_id);
    bool              any_name = false;
    size_t            start    = 0;
    while (start <= device_str.size()) {
        size_t            end  = device_str.find(',', start);
        const std::string name = device_str.substr(start, end == std::string::npos ? end : end - start);
        start                  = (end == std::string::npos) ? device_str.size() + 1 : end + 1;
        if (name.empty()) {
            continue;
        }
        any_name = true;

        auto* dev = ggml_backend_dev_by_name(name.c_str());
        if (!dev) {
            GENIEX_LOG_WARN("Device '{}' not found, skipping", name);
            continue;
        }
        devices.push_back(dev);
        GENIEX_LOG_INFO("Found device: {}", name);
    }

    if (any_name && devices.empty()) {
        GENIEX_LOG_ERROR("No valid devices found in '{}'", device_id);
        return std::nullopt;
    }
    if (!devices.empty()) {
        GENIEX_LOG_INFO("Using {} device(s): {}", devices.size(), device_id);
        devices.push_back(nullptr);  // NULL terminator for llama_model_params::devices
    }
    return devices;
}

// IQ9: HTP0 row-splits across both NSPs (small models, mmproj), HTP1-HTP3 are single sessions on NSP0 and HTP4-HTP6
// on NSP1, for layer-splitting models too large for one session.
constexpr const char* kIq9HexagonDevices = "HTP0[0-1:0],HTP1[0:1],HTP2[0:2],HTP3[0:3],HTP4[1:1],HTP5[1:2],HTP6[1:3]";

const char* linux_hexagon_devices() {
    if (!std::filesystem::exists("/dev/fastrpc-cdsp1")) return "2";
    char* chipset = nullptr;
    geniex_model_detect_chipset(/*offline=*/1, &chipset);
    const bool iq9 = chipset && std::string_view(chipset) == "qualcomm-qcs9075";
    if (chipset) geniex_free(chipset);
    return iq9 ? kIq9HexagonDevices : "HTP0[0-1:0],HTP1[0-1:1]";
}

void select_iq9_npu_devices(std::vector<ggml_backend_dev_t>& selection, const char* model_path) {
    const char* env = std::getenv("GGML_HEXAGON_DEVICES");
    if (!env || std::strcmp(env, kIq9HexagonDevices) != 0 || !model_path) return;
    // resolve_devices() output is null-terminated: plain `npu` is {HTP0, nullptr}.
    if (selection.size() != 2 || !selection[0] || std::strcmp(ggml_backend_dev_name(selection[0]), "HTP0") != 0) return;

    std::error_code ec;
    const uintmax_t size = std::filesystem::file_size(model_path, ec);
    if (ec || size < (2ull << 30)) return;
    // Keep the per-NSP grouping: alternating cdsp/cdsp1 between consecutive layers aborts with dsp-rsp
    // INTERNAL-ERROR. Up to ~8 GiB (4B-13B Q4) two sessions per NSP fit and leave HTP0 to the mmproj; larger
    // models (e.g. Qwen3-30B-A3B) need all seven.
    const char* devices = size >= (8ull << 30) ? "HTP0,HTP1,HTP2,HTP3,HTP4,HTP5,HTP6" : "HTP1,HTP2,HTP4,HTP5";
    auto        split   = resolve_devices(devices);
    if (!split || split->empty()) return;
    GENIEX_LOG_INFO("IQ9 NPU: {} MiB model, layer-splitting across {}", size >> 20, devices);
    selection = std::move(*split);
}

void apply_device_selection(llama_model_params& mpar, std::vector<ggml_backend_dev_t>& selection, Device device) {
    static ggml_backend_dev_t no_devices[] = {nullptr};
    if (selection.empty()) {
        if (device == Device::CPU) {
            mpar.devices = no_devices;
            return;
        }
        // Mirror llama.cpp's default pick (llama.cpp: "default device selection"): non-RPC GPU devices, plus iGPUs only
        // when there is no GPU, but keep just the first HTP.
        bool                            seen_htp    = false;
        bool                            skipped_htp = false;
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev  = ggml_backend_dev_get(i);
            const auto         type = ggml_backend_dev_type(dev);
            if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) continue;
            if (std::string_view(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) == "RPC") continue;
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU && std::strncmp(ggml_backend_dev_name(dev), "HTP", 3) == 0) {
                if (seen_htp) {
                    skipped_htp = true;
                    continue;
                }
                seen_htp = true;
            }
            (type == GGML_BACKEND_DEVICE_TYPE_GPU ? gpus : igpus).push_back(dev);
        }
        std::vector<ggml_backend_dev_t> defaults = gpus.empty() ? igpus : gpus;
        if (!skipped_htp) return;  // at most one HTP: llama.cpp's own default is already what we want.
        GENIEX_LOG_INFO("Default device list restricted to {} device(s); spare HTP sessions are left to the mmproj",
            defaults.size());
        selection = std::move(defaults);
        selection.push_back(nullptr);
    }
    mpar.devices = selection.data();
}

ggml_backend_dev_t resolve_vision_device(const std::vector<ggml_backend_dev_t>& lm_devices) {
    if (lm_devices.empty()) return nullptr;

    auto is_htp = [](ggml_backend_dev_t dev) {
        const char* name = dev ? ggml_backend_dev_name(dev) : nullptr;
        return name && std::strncmp(name, "HTP", 3) == 0;
    };

    if (!is_htp(lm_devices.front())) {
        return lm_devices.front();  // GPU/CPU: no session-contention concern, keep as-is.
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!is_htp(dev)) continue;
        bool used_by_lm =
            std::any_of(lm_devices.begin(), lm_devices.end(), [&](ggml_backend_dev_t d) { return d == dev; });
        if (!used_by_lm) {
            GENIEX_LOG_INFO("Using separate HTP session '{}' for mmproj", ggml_backend_dev_name(dev));
            return dev;
        }
    }

    GENIEX_LOG_WARN(
        "No spare HTP session for mmproj (GGML_HEXAGON_DEVICES=1); sharing '{}' with the LM, which may fail to "
        "allocate its compute buffer",
        ggml_backend_dev_name(lm_devices.front()));
    return lm_devices.front();
}

void apply_tool_fields(common_chat_msg& msg, const geniex_ToolCall* tool_calls, int32_t tool_call_count,
    const char* tool_call_id, const char* tool_name) {
    if (tool_calls && tool_call_count > 0) {
        msg.tool_calls.reserve(static_cast<size_t>(tool_call_count));
        for (int32_t i = 0; i < tool_call_count; ++i) {
            const geniex_ToolCall& src = tool_calls[i];
            if (!src.name) {
                GENIEX_LOG_WARN("Skipping tool call {} with null name", i);
                continue;
            }
            common_chat_tool_call tc;
            tc.name      = src.name;
            tc.arguments = src.arguments ? src.arguments : "{}";
            tc.id        = src.id ? src.id : "";
            msg.tool_calls.push_back(std::move(tc));
        }
    }
    if (tool_call_id) msg.tool_call_id = tool_call_id;
    if (tool_name) msg.tool_name = tool_name;
}

}  // namespace geniex
