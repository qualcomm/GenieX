#pragma once

#include <tokenizers_cpp.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "nexaproc-export.h"

// Constants
#define GENIEX_DEFAULT_SEED UINT32_MAX

#define GENIEX_TOKEN_NULL -1

// Core types
typedef int32_t geniex_token;

// Vocab interface for grammar support (optional)
struct nexa_vocab_interface;

GENIEXPROC_API nexa_vocab_interface* create_nexa_vocab_tokenizers(const std::string& vocab_path);
GENIEXPROC_API nexa_vocab_interface* create_nexa_vocab_tokenizers(tokenizers::Tokenizer* tokenizer);

// Sampler

enum GENIEXPROC_API sampler_type {
    SAMPLER_TYPE_NONE  = 0,
    SAMPLER_TYPE_DRY   = 1,
    SAMPLER_TYPE_TOP_K = 2,
    SAMPLER_TYPE_TOP_P = 3,
    SAMPLER_TYPE_MIN_P = 4,
    // SAMPLER_TYPE_TFS_Z       = 5,
    SAMPLER_TYPE_TYPICAL_P   = 6,
    SAMPLER_TYPE_TEMPERATURE = 7,
    SAMPLER_TYPE_XTC         = 8,
    SAMPLER_TYPE_INFILL      = 9,
    SAMPLER_TYPE_PENALTIES   = 10,
    SAMPLER_TYPE_TOP_N_SIGMA = 11,
};

// Sampling parameters (similar to common_params_sampling)
struct GENIEXPROC_API nexa_sampler_params {
    uint32_t seed = GENIEX_DEFAULT_SEED;

    int32_t n_prev    = 64;     // number of previous tokens to remember
    int32_t top_k     = 40;     // <= 0 to use vocab size
    float   top_p     = 0.95f;  // 1.0 = disabled
    float   min_p     = 0.05f;  // 0.0 = disabled
    float   temp      = 0.80f;  // <= 0.0 to sample greedily
    float   typical_p = 1.00f;  // 1.0 = disabled
    size_t  min_keep  = 0;      // minimum number of tokens to keep

    // Penalties
    int32_t penalty_last_n  = 64;     // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float   penalty_repeat  = 1.00f;  // 1.0 = disabled
    float   penalty_freq    = 0.00f;  // 0.0 = disabled
    float   penalty_present = 0.00f;  // 0.0 = disabled

    // DRY sampling
    float                    dry_multiplier        = 0.0f;  // 0.0 = disabled
    float                    dry_base              = 1.75f;
    int32_t                  dry_allowed_length    = 2;
    int32_t                  dry_penalty_last_n    = -1;                      // -1 = context size
    std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"};  // default sequence breakers for DRY

    // XTC sampling
    float xtc_probability = 0.00f;  // 0.0 = disabled
    float xtc_threshold   = 0.10f;

    // Mirostat
    int32_t mirostat     = 0;  // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float   mirostat_tau = 5.00f;
    float   mirostat_eta = 0.10f;

    // Extended temperature
    float dynatemp_range    = 0.00f;  // 0.0 = disabled
    float dynatemp_exponent = 1.00f;

    // Top-N-Sigma sampling
    float top_n_sigma = -1.00f;  // -1.0 = disabled

    // Grammar (requires vocab interface)
    std::string grammar_str;
    std::string grammar_root = "root";

    // Performance
    bool no_perf = false;  // disable performance measurements

    // Logit bias
    std::vector<std::pair<geniex_token, float>> logit_bias;

    std::vector<enum sampler_type> samplers = {
        SAMPLER_TYPE_PENALTIES,
        SAMPLER_TYPE_DRY,
        SAMPLER_TYPE_TOP_N_SIGMA,
        SAMPLER_TYPE_TOP_K,
        SAMPLER_TYPE_TYPICAL_P,
        SAMPLER_TYPE_TOP_P,
        SAMPLER_TYPE_MIN_P,
        SAMPLER_TYPE_XTC,
        SAMPLER_TYPE_TEMPERATURE,
    };

    // print the parameters into a string
    std::string print() const;
};

// High-level sampler context (contains both grammar and chain)
//
// Usage pattern (similar to common_sampler):
//   // Initialize parameters
//   nexa_sampler_params params;
//   params.temp = 0.7f;
//   params.top_k = 40;
//
//   // Initialize context
//   auto* sctx = nexa_sampler_init_context(params, vocab);
//
//   // In sampling loop:
//   nexa_sampler_context_set_logits(sctx, model_logits, vocab_size);
//   auto token = nexa_sampler_context_sample(sctx);
//   nexa_sampler_context_accept(sctx, token);
//
//   // Print performance
//   nexa_perf_sampler_context_print(sctx);
//
//   // Free the context
//   nexa_sampler_context_free(sctx);
//

// C-style interface
// Forward declaration
struct nexa_sampler_context;

GENIEXPROC_API struct nexa_sampler_context* nexa_sampler_init_context(
    const nexa_sampler_params& params, nexa_vocab_interface* vocab);
GENIEXPROC_API struct nexa_sampler_context* nexa_sampler_context_clone(const struct nexa_sampler_context* sctx);
GENIEXPROC_API std::string nexa_sampler_context_print(const struct nexa_sampler_context* sctx);
GENIEXPROC_API void        nexa_sampler_context_set_logits(
           struct nexa_sampler_context* sctx, const float* logits, int32_t n_vocab);
GENIEXPROC_API geniex_token nexa_sampler_context_sample(
    struct nexa_sampler_context* sctx, const float* logits, int32_t n_vocab, bool grammar_first = false);
GENIEXPROC_API geniex_token nexa_sampler_context_sample_no_accept(
    struct nexa_sampler_context* sctx, const float* logits, int32_t n_vocab, bool grammar_first = false);
GENIEXPROC_API void nexa_sampler_context_accept(struct nexa_sampler_context* sctx, geniex_token token);
GENIEXPROC_API void nexa_sampler_context_reset(struct nexa_sampler_context* sctx);
GENIEXPROC_API void nexa_sampler_context_free(struct nexa_sampler_context* sctx);
GENIEXPROC_API void nexa_perf_sampler_context_print(const struct nexa_sampler_context* sctx);

// Access the current candidate tokens (after set_logits, before/after sampling)
// forward declaration
struct geniex_token_data_array;
typedef geniex_token_data_array* geniex_token_data_array_t;

GENIEXPROC_API geniex_token_data_array_t nexa_sampler_context_get_candidates(struct nexa_sampler_context* sctx);
// End of C-style interface

// OOP interface
class GENIEXPROC_API NexaSampler {
   public:
    NexaSampler(const nexa_sampler_params& params = nexa_sampler_params(), tokenizers::Tokenizer* tokenizer = nullptr);
    ~NexaSampler();

    geniex_token        sample(const std::vector<float>& logits, bool grammar_first = false);
    static geniex_token sample_greedy(const std::vector<float>& logits) {
        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }
    void reset();
    void init(const std::vector<int32_t>& input_ids);

    // only works with vocab set
    bool is_eog_token(geniex_token token) const;

    void print_chain() const;
    void print_perf() const;

   private:
    nexa_sampler_params                   params_;
    std::unique_ptr<nexa_vocab_interface> vocab_;
    std::unique_ptr<nexa_sampler_context> sctx_;
};
// End of OOP interface