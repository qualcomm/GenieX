#include "params.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

#include "logging.h"

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
    if (n_gpu_layers <= 0) return Device::CPU;
    if (!device_id || device_id[0] == '\0') return Device::NPU;  // HYBRID treat as NPU
    const std::string id(device_id);
    if (id.rfind("GPU", 0) == 0) return Device::GPU;
    if (id.rfind("HTP", 0) == 0) return Device::NPU;
    return Device::CPU;  // UNKOWN treat as CPU
}

int resolve_n_threads(int requested, Device device) {
    static const int cpu_threads      = static_cast<int32_t>(std::thread::hardware_concurrency());
    static const int cpu_matrix[3][3] = {
        {cpu_threads, 6, 6},  // Linux
        {cpu_threads, 6, 6},  // Windows
        {cpu_threads, 6, 6}   // Android
    };

    if (requested > 0) return requested;

    return cpu_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];
}

llama_model_params build_model_params(const geniex_ModelConfig& config, Device device) {
    static const bool mmap_matrix[3][3] = {
        {true, false, false},   // Linux
        {false, false, false},  // Windows
        {false, false, false}   // Android
    };

    bool use_mmap = mmap_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    llama_model_params mpar = llama_model_default_params();
    mpar.use_mmap           = use_mmap;
    mpar.use_mlock          = false;
    mpar.n_gpu_layers       = config.n_gpu_layers;
    GENIEX_LOG_INFO("[Optimise] model params: n_gpu_layers={}, use_mmap={}, use_mlock={}",
        mpar.n_gpu_layers,
        mpar.use_mmap,
        mpar.use_mlock);
    return mpar;
}

llama_context_params build_context_params(const geniex_ModelConfig& config, int32_t n_ctx_default, Device device) {
    static const uint32_t ubatch_matrix[3][3] = {
        {2048, 2048, 2048},  // Linux
        {2048, 512, 1024},   // Windows
        {2048, 2048, 2048}   // Android
    };
    static const bool fa_matrix[3][3] = {
        {true, true, true},   // Linux
        {true, false, true},  // Windows
        {true, true, true}    // Android
    };

    uint32_t ubatch = ubatch_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];
    bool     fa     = fa_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    llama_context_params cpar = llama_context_default_params();
    cpar.n_ctx                = config.n_ctx > 0 ? config.n_ctx : n_ctx_default;
    cpar.n_batch              = config.n_batch > 0 ? config.n_batch : 2048;
    cpar.n_ubatch             = config.n_ubatch > 0 ? config.n_ubatch : ubatch;
    cpar.n_seq_max            = config.n_seq_max > 0 ? config.n_seq_max : 1;
    cpar.n_threads            = resolve_n_threads(config.n_threads, device);
    cpar.n_threads_batch      = resolve_n_threads(config.n_threads_batch, device);
    cpar.flash_attn_type      = static_cast<llama_flash_attn_type>(fa);
    cpar.no_perf              = false;

    GENIEX_LOG_INFO(
        "[Optimise] context params: n_ctx={}, n_batch={}, n_ubatch={}, n_seq_max={}, n_threads={}, "
        "n_threads_batch={}, flash_attn_type={}, swa_full={}, kv_unified={}, no_perf={}",
        cpar.n_ctx,
        cpar.n_batch,
        cpar.n_ubatch,
        cpar.n_seq_max,
        cpar.n_threads,
        cpar.n_threads_batch,
        static_cast<int>(cpar.flash_attn_type),
        cpar.swa_full,
        cpar.kv_unified,
        cpar.no_perf);
    return cpar;
}

ggml_threadpool_params build_threadpool_params(int n_threads, Device device) {
    static const bool pin_matrix[3][3] = {
        {false, true, true},    // Linux
        {false, false, false},  // Windows
        {false, true, true}     // Android
    };
    static const uint32_t poll_matrix[3][3] = {
        {0, 1000, 1000},  // Linux
        {0, 0, 0},        // Windows
        {0, 1000, 1000}   // Android
    };

    int      reserved_cores = 2;
    uint32_t poll           = poll_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];
    bool     pin            = pin_matrix[static_cast<int>(kHostPlatform)][static_cast<int>(device)];

    ggml_threadpool_params tpp = ggml_threadpool_params_default(n_threads);

    if (pin) {
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

    return s;
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

}  // namespace geniex
