// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Single source of truth for the user-facing device alias table
// (cpu / gpu / npu / hybrid → concrete device_id + n_gpu_layers).
// Language bindings (Go CLI, Python, Android/JNI) call through to this.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include "geniex.h"
#include "logging.h"

#if defined(_WIN32)
#define portable_strdup _strdup
#else
#define portable_strdup strdup
#endif

namespace {

constexpr const char* kPluginQairt = "qairt";

constexpr const char* kAliasCPU    = "cpu";
constexpr const char* kAliasGPU    = "gpu";
constexpr const char* kAliasNPU    = "npu";
constexpr const char* kAliasHybrid = "hybrid";
constexpr const char* kAliasAuto   = "auto";

constexpr const char* kDeviceHTP0      = "HTP0";
constexpr const char* kDeviceGPUOpenCL = "GPUOpenCL";
constexpr const char* kDeviceQairtNPU  = "NPU";

std::string to_lower(const char* s) {
    if (!s) return {};
    std::string out(s);
    std::transform(
        out.begin(), out.end(), out.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(start, end - start);
}

std::string to_lower_trim(const char* s) { return trim(to_lower(s)); }

bool is_known_alias(const std::string& s) {
    return s == kAliasCPU || s == kAliasGPU || s == kAliasNPU || s == kAliasHybrid;
}

// A concrete llama.cpp device name: "HTP0".."HTPn" or "GPUOpenCL".
bool is_device_token(const std::string& t) {
    if (t == kDeviceGPUOpenCL) return true;
    if (t.rfind("HTP", 0) == 0 && t.size() > 3) {
        for (size_t i = 3; i < t.size(); ++i)
            if (!std::isdigit(static_cast<unsigned char>(t[i]))) return false;
        return true;
    }
    return false;
}

// Parse an explicit device list like "HTP0,HTP1,HTP2,HTP3" (or a single
// "HTP0"). Returns true and fills `out` with the whitespace-stripped list
// only when every comma-separated token is a valid device name.
bool parse_device_list(const std::string& raw, std::string& out) {
    if (raw.empty()) return false;
    std::string normalized;
    size_t      start = 0;
    while (start <= raw.size()) {
        size_t      end = raw.find(',', start);
        std::string tok = trim(raw.substr(start, end == std::string::npos ? std::string::npos : end - start));
        start           = (end == std::string::npos) ? raw.size() + 1 : end + 1;
        if (!is_device_token(tok)) return false;
        if (!normalized.empty()) normalized += ',';
        normalized += tok;
    }
    out = normalized;
    return !out.empty();
}

}  // namespace

int32_t geniex_resolve_device(const geniex_ResolveDeviceInput* input, geniex_ResolveDeviceOutput* output) {
    if (!input || !output) {
        GENIEX_LOG_ERROR("geniex_resolve_device: input/output is null");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    // Initialise output so partial failures leave a sane state.
    output->device_id = nullptr;
    output->ngl       = input->ngl_default;
    output->warning   = nullptr;

    if (!input->plugin_id) {
        GENIEX_LOG_ERROR("geniex_resolve_device: plugin_id is null");
        return GENIEX_ERROR_COMMON_INVALID_INPUT;
    }

    const std::string plugin = input->plugin_id;
    const std::string raw    = trim(input->mode ? input->mode : "");
    std::string       alias  = to_lower(raw.c_str());

    // Besides the aliases, mode may be an explicit device list ("HTP0,HTP1")
    // that bypasses the alias table and is handed straight to llama.cpp.
    std::string device_list;
    const bool  has_device_list = parse_device_list(raw, device_list);

    if (!alias.empty() && alias != kAliasAuto && !is_known_alias(alias) && !has_device_list) {
        GENIEX_LOG_ERROR("geniex_resolve_device: invalid device mode '{}'", raw);
        return GENIEX_ERROR_COMMON_INVALID_DEVICE;
    }

    // Empty / "auto" → plugin default. Both qairt and llama_cpp default to
    // the pinned-NPU path.
    if (alias.empty() || alias == kAliasAuto) {
        alias = kAliasNPU;
    }

    // QAIRT is NPU-only and rejects any non-zero n_gpu_layers, so force
    // ngl to 0. Non-npu aliases and device lists are coerced with a
    // warning, not an error.
    if (plugin == kPluginQairt) {
        if (has_device_list || alias != kAliasNPU) {
            const std::string shown = has_device_list ? device_list : alias;
            std::string       msg =
                "qairt plugin only supports NPU inference; ignoring device='" + shown + "' and running on NPU";
            output->warning = portable_strdup(msg.c_str());
        }
        output->device_id = portable_strdup(kDeviceQairtNPU);
        output->ngl       = 0;
        return GENIEX_SUCCESS;
    }

    // llama_cpp: an explicit device list passes through verbatim; ngl is
    // left at its default (-1 = "all layers").
    if (has_device_list) {
        output->device_id = portable_strdup(device_list.c_str());
        return GENIEX_SUCCESS;
    }

    // ngl passes through unchanged (-1 means "all layers" to llama.cpp).
    // Only cpu forces it to 0.
    if (alias == kAliasCPU) {
        output->ngl = 0;
    } else if (alias == kAliasGPU) {
        output->device_id = portable_strdup(kDeviceGPUOpenCL);
    } else if (alias == kAliasNPU) {
        output->device_id = portable_strdup(kDeviceHTP0);
    }
    return GENIEX_SUCCESS;
}
