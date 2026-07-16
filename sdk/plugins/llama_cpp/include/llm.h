// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "htp_session.h"
#include "llama.h"
#include "params.h"
#include "plugin/ILlm.h"
#include "profiler.h"
#include "sampling.h"
#include "speculative.h"
#include "threadpool.h"

namespace geniex {

class LlamaLlm : public ILlm {
    llama_model*               model   = nullptr;
    llama_context*             ctx     = nullptr;
    common_sampler*            sampler = nullptr;
    Threadpools                pools_;
    std::optional<std::string> chat_template_str = std::nullopt;

    // Speculative decoding (MTP). All null unless a draft model was loaded.
    llama_model*        draft_model = nullptr;
    llama_context*      draft_ctx   = nullptr;
    common_speculative* spec        = nullptr;
    int32_t             spec_n_max  = 0;

    int                      n_past_global = 0;
    int                      n_past        = 0;   // for context shifting
    std::vector<llama_token> past_prompt_tokens;  // for prefix match

    bool allow_special_tokens = false;  // Control special token output

    // Tracks whether this instance pinned an HTP session; releases on last handoff.
    htp::SessionGuard htp_guard_;

   public:
    virtual ~LlamaLlm() override;

    virtual int32_t create_impl(const geniex_LlmCreateInput*) override;

    virtual int32_t reset() override;

    virtual int32_t save_kv_cache(const geniex_KvCacheSaveInput*, geniex_KvCacheSaveOutput*) override;
    virtual int32_t load_kv_cache(const geniex_KvCacheLoadInput*, geniex_KvCacheLoadOutput*) override;

    virtual int32_t apply_chat_template(
        const geniex_LlmApplyChatTemplateInput*, geniex_LlmApplyChatTemplateOutput*) override;

    virtual int32_t generate(const geniex_LlmGenerateInput*, geniex_LlmGenerateOutput*) override;

    virtual int32_t get_model_info(geniex_LlmModelInfo*) override;

   private:
    void set_sampler(const geniex_SamplerConfig* cfg);

    int32_t setup_speculative(const geniex_ModelConfig& config, Device device, const char* device_id);
    void    teardown_speculative();
    int32_t decode_speculative(const geniex_GenerationConfig& cfg, const std::vector<llama_token>& prompt_ids,
        const std::function<bool(llama_token)>& emit, const std::function<int()>& n_generated,
        ::common::Profiler& profiler);
};

}  // namespace geniex
