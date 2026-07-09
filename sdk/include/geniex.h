// Copyright (c) 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

/**
 * @file geniex.h
 * @brief Unified C API for machine learning operations
 *
 * This header provides a C interface for language models (LLM) and multimodal
 * models (VLM).
 *
 * All functions return status codes where applicable, with negative values indicating errors.
 * Memory management follows RAII principles - use corresponding destroy/free functions.
 */

#include <stdbool.h>
#include <stdint.h>

#ifdef GENIEX_SHARED
#if defined(_WIN32) && !defined(__MINGW32__)
#ifdef GENIEX_BUILD
#define GENIEX_API __declspec(dllexport)
#else
#define GENIEX_API __declspec(dllimport)
#endif
#else
#define GENIEX_API __attribute__((visibility("default")))
#endif
#else
#define GENIEX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Error code enumeration for ML operations */
typedef enum {
    /** ===== SUCCESS ===== */

    GENIEX_SUCCESS = 0, /**< Operation completed successfully */

    /* ===== COMMON ERRORS (100xxx) ===== */

    GENIEX_ERROR_COMMON_UNKNOWN             = -100000, /**< Unknown error */
    GENIEX_ERROR_COMMON_INVALID_INPUT       = -100001, /**< Invalid input parameters or handle */
    GENIEX_ERROR_COMMON_INVALID_DEVICE      = -100002, /**< Unknown device alias (cpu/gpu/npu/hybrid) */
    GENIEX_ERROR_COMMON_MEMORY_ALLOCATION   = -100003, /**< Memory allocation failed */
    GENIEX_ERROR_COMMON_FILE_NOT_FOUND      = -100004, /**< File not found or inaccessible */
    GENIEX_ERROR_COMMON_NETWORK             = -100005, /**< Network failure (timeout, bad status, DNS, proxy, ...) */
    GENIEX_ERROR_COMMON_CANCELLED           = -100006, /**< Operation cancelled by caller */
    GENIEX_ERROR_COMMON_NOT_INITIALIZED     = -100007, /**< Library not initialized */
    GENIEX_ERROR_COMMON_ALREADY_INITIALIZED = -100008, /**< Library already initialized; deinit first */
    GENIEX_ERROR_COMMON_AUTH                = -100009, /**< Hub rejected request; auth required (HTTP 401/403) */
    GENIEX_ERROR_COMMON_HUB_MODEL_NOT_FOUND = -100010, /**< Model not found on the remote hub (HTTP 404) */
    GENIEX_ERROR_COMMON_RATE_LIMITED        = -100011, /**< Hub rate limit exceeded (HTTP 429) */
    GENIEX_ERROR_COMMON_HUB_SERVER          = -100012, /**< Hub server error (HTTP 5xx) */
    GENIEX_ERROR_COMMON_NOT_SUPPORTED       = -100013, /**< Operation not supported */
    GENIEX_ERROR_COMMON_MANIFEST_PARSE      = -100014, /**< Failed to parse a manifest / index document */
    GENIEX_ERROR_COMMON_CHIPSET_UNAVAILABLE = -100015, /**< Requested chipset not available for this model */
    GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED = -100016, /**< Parameter not supported by this plugin */

    GENIEX_ERROR_COMMON_MODEL_LOAD    = -100201, /**< Model loading failed */
    GENIEX_ERROR_COMMON_MODEL_INVALID = -100203, /**< Invalid model format */

    GENIEX_ERROR_COMMON_PLUGIN_LOAD    = -100301, /**< Plugin loading failed */
    GENIEX_ERROR_COMMON_PLUGIN_INVALID = -100302, /**< Invalid plugin */

    GENIEX_ERROR_COMMON_LICENSE_INVALID = -100601, /**< Invalid license */
    GENIEX_ERROR_COMMON_LICENSE_EXPIRED = -100602, /**< License expired */

    /* ===== LLM ERRORS (200xxx) ===== */

    GENIEX_ERROR_LLM_TOKENIZATION_FAILED         = -200001, /**< Tokenization failed */
    GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH = -200004, /**< Context length exceeded */

    GENIEX_ERROR_LLM_GENERATION_FAILED          = -200101, /**< Text generation failed */
    GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG = -200103, /**< Input prompt too long */

    /* ===== VLM ERRORS (201xxx) ===== */

    GENIEX_ERROR_VLM_IMAGE_LOAD   = -201001, /**< Image loading failed */
    GENIEX_ERROR_VLM_IMAGE_FORMAT = -201002, /**< Unsupported image format */

    GENIEX_ERROR_VLM_AUDIO_LOAD   = -201101, /**< Audio loading failed */
    GENIEX_ERROR_VLM_AUDIO_FORMAT = -201102, /**< Unsupported audio format */

    GENIEX_ERROR_VLM_GENERATION_FAILED = -201201, /**< Multimodal generation failed */

} geniex_ErrorCode;

/** Get error message string for error code */
GENIEX_API const char* geniex_get_error_message(const geniex_ErrorCode error_code);

/* ========================================================================== */
/*                              CORE TYPES & UTILITIES                         */
/* ========================================================================== */

/** Plugin Id string type - plain char* for plugin id
 * @ref geniex_get_device_list device must in list of plugin ids
 */
typedef const char* geniex_PluginId;

/** Path string type - plain char* for file paths */
typedef const char* geniex_Path;

typedef enum {
    GENIEX_LOG_LEVEL_TRACE, /* Trace messages */
    GENIEX_LOG_LEVEL_DEBUG, /* Debug messages */
    GENIEX_LOG_LEVEL_INFO,  /* Informational messages */
    GENIEX_LOG_LEVEL_WARN,  /* Warning messages */
    GENIEX_LOG_LEVEL_ERROR  /* Error messages */
} geniex_LogLevel;

/** Logging callback function type */
typedef void (*geniex_log_callback)(geniex_LogLevel, const char*);

/** Token callback for streaming generation */
typedef bool (*geniex_token_callback)(const char* token, void* user_data);

/** Input structure for saving KV cache */
typedef struct {
    geniex_Path path; /** Path to save the KV cache */
} geniex_KvCacheSaveInput;

/** Output structure for saving KV cache (empty for now) */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} geniex_KvCacheSaveOutput;

/** Input structure for loading KV cache */
typedef struct {
    geniex_Path path; /** Path to load the KV cache from */
} geniex_KvCacheLoadInput;

/** Output structure for loading KV cache (empty for now) */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} geniex_KvCacheLoadOutput;

/* ====================  Core Initialization  ================================ */

/**
 * @brief Initialize the ML C-Lib runtime, starting the life cycle of the library.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
GENIEX_API int32_t geniex_init(void);

/** Plugin id create function type */
typedef geniex_PluginId (*geniex_plugin_id_func)();

/** Plugin instance create function type */
typedef void* (*geniex_create_plugin_func)();

/**
 * @brief Register a custom plugin with the ML C-Lib runtime.
 *
 * @param plugin_id_func[in]: The pointer to plugin create_id function.
 * @param create_func[in]: The pointer to plugin create function.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe.
 */
GENIEX_API int32_t geniex_register_plugin(geniex_plugin_id_func plugin_id_func, geniex_create_plugin_func create_func);

/**
 * @brief Deinitialize the ML C-Lib runtime, ending the life cycle of the library.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe.
 */
GENIEX_API int32_t geniex_deinit(void);

/**
 * @brief Set custom logging callback function, call before init
 *
 * @param callback[in]: The callback function to set.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe
 */
GENIEX_API int32_t geniex_set_log(geniex_log_callback callback);

/**
 * @brief Simple wrapper around free() to free memory allocated by ML library functions
 *
 * @param ptr[in]: The pointer to free.
 *
 * @thread_safety: Thread-safe if called for different pointers.
 */
GENIEX_API void geniex_free(void* ptr);

/**
 * @brief Get Library Version
 *
 * @return Null-terminated UTF-8 string.
 *
 * @thread_safety: Thread-safe.
 */
GENIEX_API const char* geniex_version(void);

/**
 * @brief Get the version string reported by a registered plugin.
 *
 * Each plugin owns its own version (e.g. the QAIRT runtime version for
 * `qairt`, the llama.cpp build commit for `llama_cpp`). The SDK bridge
 * forwards to the plugin's `Plugin::version()` override so that
 * `libgeniex` itself does not need to link against any plugin-specific
 * runtime.
 *
 * @param plugin_id[in]: Plugin identifier (must be non-NULL).
 *
 * @return Null-terminated UTF-8 string owned by the plugin. Returns NULL
 *         when `plugin_id` is NULL or no matching plugin is registered.
 *
 * @thread_safety: Thread-safe.
 */
GENIEX_API const char* geniex_get_plugin_version(geniex_PluginId plugin_id);

/** Output structure containing the list of available plugins */
typedef struct {
    geniex_PluginId* plugin_ids;   /**< Array of plugin IDs (UTF-8) (caller must free with geniex_free) */
    int32_t          plugin_count; /**< Number of plugin IDs in the list */
} geniex_GetPluginListOutput;

/**
 * @brief Query the list of available plugins.
 *
 * @param output[out] Pointer to plugin list and count. caller must free with `geniex_free`.
 *
 * @return geniex_ErrorCode GENIEX_SUCCESS on success, negative value on failure.
 *
 * @thread_safety: Not thread-safe.
 *
 * @note The returned plugin_list TODO
 */
GENIEX_API int32_t geniex_get_plugin_list(geniex_GetPluginListOutput* output);

/** Input structure for querying available devices for a plugin */
typedef struct {
    geniex_PluginId plugin_id; /**< Plugin identifier */
} geniex_GetDeviceListInput;

/** Output structure containing the list of available devices */
typedef struct {
    // example: HTP0
    const char** device_ids;   /**< Array of device IDs  (caller must free with geniex_free when not null) */
    const char** device_names; /**< Array of device names  (caller must free with geniex_free when not null) */
    int32_t      device_count; /**< Number of device names in the list */
} geniex_GetDeviceListOutput;

/**
 * @brief Query the list of available devices for a given plugin.
 *
 * @param input[in]   Pointer to input structure specifying the plugin.
 * @param output[out] Pointer to output structure to receive device list and count.
 *
 * @return geniex_ErrorCode GENIEX_SUCCESS on success, negative value on failure.
 *
 * @thread_safety: Not thread-safe.
 *
 * @note The returned device_list TODO
 */
GENIEX_API int32_t geniex_get_device_list(const geniex_GetDeviceListInput* input, geniex_GetDeviceListOutput* output);

/**
 * Input for geniex_resolve_device.
 *
 * `mode` is a user-facing alias: one of "cpu", "gpu", "npu", "hybrid",
 * the empty string, or "auto" (both of which mean "let the SDK pick the
 * plugin's default"). Matching is case-insensitive; surrounding whitespace
 * is trimmed.
 *
 * `model_name` is an optional hint (may be NULL) that lets the SDK apply
 * model-specific overrides to the "auto" default — e.g. `llama_cpp` models
 * whose name contains `gpt-oss` default to `npu` instead of `hybrid`
 * because the hybrid per-tensor scheduler can't place all their ops on
 * HTP end-to-end.
 *
 * `ngl_default` is forwarded as the caller's preferred `n_gpu_layers`
 * value. The resolver returns it unchanged in `ngl` except when the
 * alias forces a specific value (cpu → 0, hybrid → 999).
 */
typedef struct {
    geniex_PluginId plugin_id;   /**< Plugin identifier (must be non-NULL) */
    const char*     model_name;  /**< Optional model name hint; NULL is OK */
    const char*     mode;        /**< User-facing alias; NULL / "" / "auto" → plugin default */
    int32_t         ngl_default; /**< Caller's default n_gpu_layers */
} geniex_ResolveDeviceInput;

/**
 * Output for geniex_resolve_device.
 *
 * `device_id` is the concrete string the SDK plugins expect (e.g.
 * "HTP0", "GPUOpenCL", "NPU"). It may be NULL, meaning "don't set
 * device_id on the create input" (the `cpu` and `hybrid` aliases for
 * llama_cpp). Callers that set the value onto a plugin input must copy
 * it and then free the output with `geniex_free` (see memory notes).
 *
 * `ngl` is the resolved `n_gpu_layers` value, already adjusted for the
 * alias (cpu → 0, hybrid → 999, otherwise `ngl_default` passes through).
 *
 * `warning` is non-NULL when the alias was coerced (e.g. qairt only has
 * an NPU device, so cpu/gpu/hybrid fall back to NPU with a warning).
 * Callers should surface the warning and continue — geniex_resolve_device
 * never returns an error for coerced modes.
 *
 * An error (GENIEX_ERROR_COMMON_INVALID_DEVICE) is returned only when
 * `mode` is a non-empty, non-"auto" string that is not one of the
 * documented aliases. Both `device_id` and `warning` are heap-allocated;
 * call `geniex_free` on each non-NULL pointer when done.
 */
typedef struct {
    char*   device_id; /**< Resolved device id; may be NULL (caller must free with geniex_free) */
    int32_t ngl;       /**< Resolved n_gpu_layers */
    char*   warning;   /**< Optional coercion warning; may be NULL (caller must free with geniex_free) */
} geniex_ResolveDeviceOutput;

/**
 * @brief Resolve a user-facing device alias into the concrete
 *        (device_id, n_gpu_layers) pair the SDK plugins expect.
 *
 * This is the single source of truth for the cpu / gpu / npu / hybrid
 * alias table. Language bindings (Go CLI, Python, Android/JNI) should
 * call this instead of reimplementing the mapping locally.
 *
 * @param input[in]   Non-NULL pointer to the input struct.
 * @param output[out] Non-NULL pointer to the output struct. On success,
 *                    `device_id` / `warning` may be heap-allocated and
 *                    must be freed by the caller with `geniex_free`.
 *
 * @return GENIEX_SUCCESS on success (including coerced aliases, where
 *         `warning` is populated). Returns GENIEX_ERROR_COMMON_INVALID_INPUT
 *         if `input` / `output` / `input->plugin_id` is NULL, or
 *         GENIEX_ERROR_COMMON_INVALID_DEVICE if `mode` is a non-empty
 *         string that is not a documented alias.
 *
 * @thread_safety: Thread-safe (pure function).
 */
GENIEX_API int32_t geniex_resolve_device(const geniex_ResolveDeviceInput* input, geniex_ResolveDeviceOutput* output);

/* ====================  Data Structures  ==================================== */

/** Profile data structure for performance metrics */
typedef struct {
    int64_t ttft;        /* Time to first token (us) */
    int64_t prompt_time; /* Prompt processing time (us) */
    int64_t decode_time; /* Token generation time (us) */

    int64_t prompt_tokens;    /* Number of prompt tokens */
    int64_t generated_tokens; /* Number of generated tokens */
    int64_t audio_duration;   /* Audio duration (us) */

    double prefill_speed;    /* Prefill speed (tokens/sec) */
    double decoding_speed;   /* Decoding speed (tokens/sec) */
    double real_time_factor; /* Real-Time Factor(RTF) (1.0 = real-time, >1.0 = faster, <1.0 = slower) */

    const char* stop_reason; /* Stop reason: "eos", "length", "user", "stop_sequence", "context_length" */
} geniex_ProfileData;

/* ========================================================================== */
/*                              LANGUAGE MODELS (LLM)                          */
/* ========================================================================== */

/** Text generation sampling parameters */
typedef struct {
    float       temperature;        /* Sampling temperature (0.0-2.0) */
    float       top_p;              /* Nucleus sampling parameter (0.0-1.0) */
    int32_t     top_k;              /* Top-k sampling parameter */
    float       min_p;              /* Minimum probability for nucleus sampling */
    float       repetition_penalty; /* Penalty for repeated tokens */
    float       presence_penalty;   /* Penalty for token presence */
    float       frequency_penalty;  /* Penalty for token frequency */
    int32_t     seed;               /* Random seed (-1 for random) */
    geniex_Path grammar_path;       /* Optional grammar file path */
    const char* grammar_string;     /* Optional grammar string (BNF-like format) */
    bool        enable_json;        /* Enable JSON grammar */
} geniex_SamplerConfig;

/** LLM / VLM generation configuration (IMPROVED: support multiple images and audios) */
typedef struct {
    int32_t               max_tokens;     /* Maximum tokens to generate */
    const char**          stop;           /* Array of stop sequences */
    int32_t               stop_count;     /* Number of stop sequences */
    int32_t               n_past;         /* Number of past tokens to consider */
    geniex_SamplerConfig* sampler_config; /* Advanced sampling config */
    // --- Improved multimodal support ---
    geniex_Path* image_paths;      /* Array of image paths for VLM (NULL if none) */
    int32_t      image_count;      /* Number of images */
    int32_t      image_max_length; /* Maximum length of the image */
    geniex_Path* audio_paths;      /* Array of audio paths for VLM (NULL if none) */
    int32_t      audio_count;      /* Number of audios */
    // --- Context-length overflow handling (qcom-ai-hub/geniex#1197) ---
    /* qairt only; llama_cpp ignores this (it always context-shifts). When true,
     * evicts the oldest context tokens above sliding_window_n_keep instead of
     * erroring on context-length overflow. */
    bool    sliding_window;
    int32_t sliding_window_n_keep; /* Tokens to keep anchored when sliding (0 = plugin default of 4) */
} geniex_GenerationConfig;

/** LLM / VLM model configuration */
typedef struct {
    int32_t n_ctx;            // text context, 0 = from model
    int32_t n_threads;        // number of threads to use for generation
    int32_t n_threads_batch;  // number of threads to use for batch processing
    int32_t n_batch;          // logical maximum batch size that can be submitted to llama_decode
    int32_t n_ubatch;         // physical maximum batch size
    int32_t n_seq_max;        // max number of sequences (i.e. distinct states for recurrent models)
    int32_t n_gpu_layers;     // number of layers to offload to GPU, 0 = all layers on CPU

    // TODO: consider removing the following fields from ModelConfig, or move to another struct
    geniex_Path chat_template_path;     // path to chat template file, optional
    const char* chat_template_content;  // content of chat template file, optional
    const char* system_prompt;          // system prompt for chat template, optional
    bool        enable_sampling;        // DEPRECATED, use enable_json in geniex_SamplerConfig
    const char* grammar_str;            // grammar string
    int32_t     max_tokens;             // max tokens to generate
    bool        enable_thinking;        // enable thinking mode for Qwen models
    bool        verbose;                // verbose logging
} geniex_ModelConfig;

/* ====================  LLM Handle  ======================================== */
typedef struct geniex_LLM geniex_LLM; /* Opaque LLM handle */

/* ====================  Lifecycle Management  ============================== */
typedef struct {
    const char*        model_name;     /** Name of the model */
    geniex_Path        model_path;     /** Path to the model file */
    geniex_Path        tokenizer_path; /** Path to the tokenizer file */
    geniex_ModelConfig config;         /** Model configuration */
    geniex_PluginId    plugin_id;      /** plugin to use for the model */
    const char*        device_id;      /** device to use for the model, NULL for default device */
    const char* license_id;  /** licence id for loading NPU models, must be provided upon the first use of the license
                                key. null terminated string */
    const char* license_key; /** licence key for loading NPU models, null terminated string */
} geniex_LlmCreateInput;

/**
 * @brief Create and initialize an LLM instance from model files
 *
 * @param input[in]: Input parameters for the LLM creation
 * @param out_handle[out]: Pointer to the LLM handle. Must be freed with geniex_llm_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_llm_create(const geniex_LlmCreateInput* input, geniex_LLM** out_handle);

/**
 * @brief Destroy LLM instance and free associated resources
 *
 * @param handle[in]: The LLM handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_llm_destroy(geniex_LLM* handle);

/**
 * @brief Reset LLM internal state (clear KV cache, reset sampling)
 *
 * @param handle[in]: The LLM handle to reset.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_llm_reset(geniex_LLM* handle);

/* ====================  KV-Cache Management  ============================== */

/**
 * @brief Save current KV cache state to file
 *
 * @param handle[in]: LLM handle
 * @param input[in]: Input parameters for saving KV cache
 * @param output[out]: Reserved struct for future use, safe to pass nullptr now
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_llm_save_kv_cache(
    geniex_LLM* handle, const geniex_KvCacheSaveInput* input, geniex_KvCacheSaveOutput* output);

/**
 * @brief Load KV cache state from file
 *
 * @param handle[in]: LLM handle
 * @param input[in]: Input parameters for loading KV cache
 * @param output[out]: Reserved struct for future use, safe to pass nullptr now
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_llm_load_kv_cache(
    geniex_LLM* handle, const geniex_KvCacheLoadInput* input, geniex_KvCacheLoadOutput* output);

/* ====================  Chat Template ================================== */

/** Chat message structure */
typedef struct {
    const char* role;    /* Message role: "user", "assistant", "system" */
    const char* content; /* Message content in UTF-8 */
} geniex_LlmChatMessage;

/** Input structure for applying chat template */
typedef struct {
    geniex_LlmChatMessage* messages;              /** Array of chat messages */
    int32_t                message_count;         /** Number of messages */
    const char*            tools;                 /** Tool JSON string (optional, can be NULL) */
    bool                   enable_thinking;       /** Enable thinking */
    bool                   add_generation_prompt; /** Add generation prompt */
} geniex_LlmApplyChatTemplateInput;

/** Output structure for applying chat template */
typedef struct {
    char* formatted_text; /** Formatted chat text (caller must free with geniex_free) */
} geniex_LlmApplyChatTemplateOutput;

/**
 * @brief Apply chat template to messages
 *
 * @param handle[in]: LLM handle
 * @param input[in]: Input parameters for applying chat template
 * @param output[out]: Output data containing the formatted text
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_llm_apply_chat_template(
    geniex_LLM* handle, const geniex_LlmApplyChatTemplateInput* input, geniex_LlmApplyChatTemplateOutput* output);

/* ====================  Streaming Generation  ============================= */

/** Input structure for streaming text generation */
typedef struct {
    const char*                    prompt_utf8; /** The full chat history as UTF-8 string */
    const geniex_GenerationConfig* config;      /** Generation configuration (optional, can be nullptr) */
    geniex_token_callback          on_token;    /** Token callback function for streaming */
    void*                          user_data;   /** User data passed to callback (optional, can be nullptr) */

    /** Mutual exclusivity rules:
     *  - If input_ids is non-NULL and input_ids_count > 0, the input_ids will be used
     *    and prompt_utf8 will be ignored.
     *  - Otherwise, prompt_utf8 must be provided.
     *  - Providing neither will result in GENIEX_ERROR_COMMON_INVALID_INPUT.
     *
     * Special tokens handling:
     *  When using input_ids, the caller is responsible for including any special tokens
     *  (BOS/EOS) as needed. The API will not add them automatically.
     */
    /** Pre-tokenized input support (alternative to prompt_utf8) */
    const int32_t* input_ids;       /** Array of pre-tokenized token IDs (optional, can be nullptr) */
    int32_t        input_ids_count; /** Number of tokens in input_ids array */
} geniex_LlmGenerateInput;

/** Output structure for streaming text generation */
typedef struct {
    char*              full_text;    /** Complete generated text (caller must free with geniex_free) */
    geniex_ProfileData profile_data; /** Profiling data for the generation */
} geniex_LlmGenerateOutput;

/**
 * @brief Generate text with streaming token callback
 *
 * @param handle[in]: LLM handle
 * @param input[in]: Input parameters for streaming generation
 * @param output[out]: Output containing the complete generated text
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_llm_generate(
    geniex_LLM* handle, const geniex_LlmGenerateInput* input, geniex_LlmGenerateOutput* output);

/* ====================  Model Metadata  =================================== */

/** Static, plugin-reported metadata about the loaded LLM. All fields are
 *  zero-initialized by the bridge before the plugin populates them, so
 *  plugins that only know a subset can leave the rest at their default. */
typedef struct {
    int32_t vocab_size; /** Number of tokens in the model vocabulary (>=1 on success). */
    int32_t bos_token;  /** BOS token id, or -1 if the model has no BOS. */
    int32_t add_bos;    /** 1 = caller should prepend BOS at position 0 when feeding raw input_ids. */
    int32_t reserved0;  /** Reserved, must be 0. */
} geniex_LlmModelInfo;

/**
 * @brief Query static metadata about the loaded LLM (vocabulary, BOS handling).
 *
 * Cheap and side-effect free; safe to call many times. Intended for callers
 * that construct raw input_ids (e.g. benchmarks doing random-id prefill) and
 * therefore need vocab_size / bos_token without going through a tokenizer.
 *
 * @param handle[in]:  LLM handle.
 * @param output[out]: Filled-in metadata. Zero-initialized before the plugin populates it.
 *
 * @return geniex_ErrorCode:
 *   - GENIEX_SUCCESS                          on success.
 *   - GENIEX_ERROR_COMMON_NOT_INITIALIZED     when handle is NULL.
 *   - GENIEX_ERROR_COMMON_INVALID_INPUT       when output is NULL.
 *   - GENIEX_ERROR_COMMON_PARAM_NOT_SUPPORTED when the plugin cannot report the metadata.
 *                                             Callers that need vocab_size must treat this
 *                                             as a hard failure.
 */
GENIEX_API int32_t geniex_llm_get_model_info(geniex_LLM* handle, geniex_LlmModelInfo* output);

/* ========================================================================== */
/*                              MULTIMODAL MODELS (VLM)                          */
/* ========================================================================== */

typedef struct {
    const char* type;  // "text", "image", "audio", … (null-terminated UTF-8)
    const char* text;  // payload: the actual text, URL, or special token
} geniex_VlmContent;

/* ---------- Message ---------- */
typedef struct {
    const char*        role;           // "user", "assistant", "system", …
    geniex_VlmContent* contents;       // dynamically-allocated array (may be NULL)
    int64_t            content_count;  // number of elements in `contents`
} geniex_VlmChatMessage;

typedef struct geniex_VLM geniex_VLM; /* Opaque VLM handle */

/* ====================  Lifecycle Management  ============================== */

typedef struct {
    const char*        model_name;     /** Name of the model */
    geniex_Path        model_path;     /** Path to the model file */
    geniex_Path        mmproj_path;    /** Path to the mmproj file */
    geniex_ModelConfig config;         /** Model configuration */
    geniex_PluginId    plugin_id;      /** Plugin to use for the model */
    const char*        device_id;      /** device to use for the model */
    geniex_Path        tokenizer_path; /** Path to the tokenizer file */
    const char* license_id;  /** licence id for loading NPU models, must be provided upon the first use of the license
                                key. null terminated string */
    const char* license_key; /** licence key for loading NPU models, null terminated string */
} geniex_VlmCreateInput;

/**
 * @brief Create and initialize a VLM instance from model files
 *
 * @param input[in]: Input parameters for the VLM creation
 * @param out_handle[out]: Pointer to the VLM handle. Must be freed with geniex_vlm_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_vlm_create(const geniex_VlmCreateInput* input, geniex_VLM** out_handle);

/**
 * @brief Destroy VLM instance and free associated resources
 *
 * @param handle[in]: The VLM handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_vlm_destroy(geniex_VLM* handle);

/**
 * @brief Reset VLM internal state (clear KV cache, reset sampling)
 *
 * @param handle[in]: The VLM handle to reset.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_vlm_reset(geniex_VLM* handle);

/* ====================  Text Generation  ================================== */

/** Input structure for applying VLM chat template */
typedef struct {
    geniex_VlmChatMessage* messages;        /** Array of chat messages */
    int32_t                message_count;   /** Number of messages */
    const char*            tools;           /** Tool JSON string (optional, can be NULL) */
    bool                   enable_thinking; /** Enable thinking */

    // deepseek-ocr
    bool grounding; /** Enable grounding (Add grounding token) */
} geniex_VlmApplyChatTemplateInput;

/** Output structure for applying VLM chat template */
typedef struct {
    char* formatted_text; /** Formatted chat text (caller must free with geniex_free) */
} geniex_VlmApplyChatTemplateOutput;

/**
 * @brief Apply chat template to messages
 *
 * @param handle[in]: VLM handle
 * @param input[in]: Input parameters for applying chat template
 * @param output[out]: Output data containing the formatted text
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_vlm_apply_chat_template(
    geniex_VLM* handle, const geniex_VlmApplyChatTemplateInput* input, geniex_VlmApplyChatTemplateOutput* output);

/* ====================  Capability Query  ================================== */

/** Modalities the loaded VLM (mmproj) reports as supported. */
typedef struct {
    bool supports_vision; /** Model can consume image inputs */
    bool supports_audio;  /** Model can consume audio inputs */
} geniex_VlmCapabilities;

/**
 * @brief Query the VLM for which input modalities its mmproj supports.
 *
 * For plugins that do not expose modality probes (e.g. QAIRT), both flags
 * default to false. For llama.cpp this reflects `mtmd_support_vision/audio`.
 *
 * @param handle[in]:  VLM handle
 * @param output[out]: Filled-in capability flags
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_vlm_get_capabilities(geniex_VLM* handle, geniex_VlmCapabilities* output);

/* ====================  Streaming Generation  ============================= */

/** Input structure for VLM streaming text generation */
typedef struct {
    const char*                    prompt_utf8; /** The full chat history as UTF-8 string */
    const geniex_GenerationConfig* config;      /** Generation configuration (optional, can be nullptr) */
    geniex_token_callback          on_token;    /** Token callback function for streaming */
    void*                          user_data;   /** User data passed to callback (optional, can be nullptr) */
} geniex_VlmGenerateInput;

/** Output structure for VLM streaming text generation */
typedef struct {
    char*              full_text;    /** Complete generated text (caller must free with geniex_free) */
    geniex_ProfileData profile_data; /** Profiling data for the generation */
} geniex_VlmGenerateOutput;

/**
 * @brief Generate text with streaming token callback
 *
 * @param handle[in]: VLM handle
 * @param input[in]: Input parameters for streaming generation
 * @param output[out]: Output containing the complete generated text
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_vlm_generate(
    geniex_VLM* handle, const geniex_VlmGenerateInput* input, geniex_VlmGenerateOutput* output);

#ifdef __cplusplus
} /* extern "C" */
#endif
