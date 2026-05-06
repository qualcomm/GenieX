#pragma once

/**
 * @file geniex.h
 * @brief Unified C API for machine learning operations
 *
 * This header provides a comprehensive C interface for various ML tasks including:
 * - Language models (LLM) and multimodal models (VLM)
 * - Text embeddings and reranking
 * - Image generation and computer vision (OCR)
 * - Speech recognition (ASR) and text-to-speech (TTS)
 * - Speaker diarization
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

    GENIEX_ERROR_COMMON_UNKNOWN           = -100000, /**< Unknown error */
    GENIEX_ERROR_COMMON_INVALID_INPUT     = -100001, /**< Invalid input parameters or handle */
    GENIEX_ERROR_COMMON_INVALID_DEVICE    = -100002, /**< Unknown device alias (cpu/gpu/npu/hybrid) */
    GENIEX_ERROR_COMMON_MEMORY_ALLOCATION = -100003, /**< Memory allocation failed */
    GENIEX_ERROR_COMMON_FILE_NOT_FOUND    = -100004, /**< File not found or inaccessible */
    GENIEX_ERROR_COMMON_NOT_INITIALIZED   = -100007, /**< Library not initialized */
    GENIEX_ERROR_COMMON_NOT_SUPPORTED     = -100013, /**< Operation not supported */

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

    /* ===== Embedding ERRORS (202xxx) ===== */

    GENIEX_ERROR_EMBEDDING_GENERATION = -202301, /**< Embedding generation failed */
    GENIEX_ERROR_EMBEDDING_DIMENSION  = -202302, /**< Invalid embedding dimension */

    /* ===== Reranking ERRORS (203xxx) ===== */

    GENIEX_ERROR_RERANK_FAILED = -203401, /**< Reranking failed */
    GENIEX_ERROR_RERANK_INPUT  = -203402, /**< Invalid reranking input */

    /* ===== Image Generation ERRORS (204xxx) ===== */

    GENIEX_ERROR_IMAGEGEN_GENERATION = -204501, /**< Image generation failed */
    GENIEX_ERROR_IMAGEGEN_PROMPT     = -204502, /**< Invalid image prompt */
    GENIEX_ERROR_IMAGEGEN_DIMENSION  = -204503, /**< Invalid image dimensions */

    /* ===== ASR ERRORS (205xxx) ===== */

    GENIEX_ERROR_ASR_TRANSCRIPTION = -205001, /**< ASR transcription failed */
    GENIEX_ERROR_ASR_AUDIO_FORMAT  = -205002, /**< Unsupported ASR audio format */
    GENIEX_ERROR_ASR_LANGUAGE      = -205003, /**< Unsupported ASR language */

    /* ===== ASR Streaming ERRORS (205xxx) ===== */

    GENIEX_ERROR_ASR_STREAM_NOT_STARTED    = -205010, /**< Streaming not started */
    GENIEX_ERROR_ASR_STREAM_ALREADY_ACTIVE = -205011, /**< Streaming already active */
    GENIEX_ERROR_ASR_STREAM_INVALID_AUDIO  = -205012, /**< Invalid audio data */
    GENIEX_ERROR_ASR_STREAM_BUFFER_FULL    = -205013, /**< Audio buffer full */
    GENIEX_ERROR_ASR_STREAM_CALLBACK_ERROR = -205014, /**< Callback execution error */

    /* ===== TTS ERRORS (206xxx) ===== */

    GENIEX_ERROR_TTS_SYNTHESIS    = -206001, /**< TTS synthesis failed */
    GENIEX_ERROR_TTS_VOICE        = -206002, /**< TTS voice not found */
    GENIEX_ERROR_TTS_AUDIO_FORMAT = -206003, /**< TTS audio format error */

    /* ===== CV ERRORS (207xxx) ===== */

    GENIEX_ERROR_CV_OCR_DETECTION   = -207001, /**< OCR text detection failed */
    GENIEX_ERROR_CV_OCR_RECOGNITION = -207002, /**< OCR text recognition failed */
    GENIEX_ERROR_CV_OCR_FAILED      = -207003, /**< OCR failed */

    /* ===== Diarization ERRORS (208xxx) ===== */

    GENIEX_ERROR_DIARIZE_AUDIO_LOAD   = -208001, /**< Audio loading failed */
    GENIEX_ERROR_DIARIZE_SEGMENTATION = -208101, /**< Segmentation model execution failed */
    GENIEX_ERROR_DIARIZE_EMBEDDING    = -208102, /**< Embedding extraction failed */
    GENIEX_ERROR_DIARIZE_CLUSTERING   = -208103, /**< Speaker clustering failed (PLDA/VBx) */

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
 * @param out_version[out]: Pointer to the library version.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe.
 */
GENIEX_API const char* geniex_version(void);

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

    const char* stop_reason; /* Stop reason: "eos", "length", "user", "stop_sequence" */
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

/* ========================================================================== */
/*                              EMBEDDING MODELS                               */
/* ========================================================================== */

/** Embedding generation configuration */
typedef struct {
    int32_t     batch_size;       /* Processing batch size */
    bool        normalize;        /* Whether to normalize embeddings */
    const char* normalize_method; /* Normalization: "l2", "mean", "none" */
} geniex_EmbeddingConfig;

typedef struct geniex_Embedder geniex_Embedder; /* Opaque embedder handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating an embedder */
typedef struct {
    const char*        model_name;     /** Name of the model */
    geniex_Path        model_path;     /** Path to the model file */
    geniex_Path        mmproj_path;    /** Path to the mmproj file */
    geniex_Path        tokenizer_path; /** Path to the tokenizer file */
    geniex_ModelConfig config;         /** Model configuration */
    geniex_PluginId    plugin_id;      /** Plugin to use for the model */
    const char*        device_id;      /** device to use for the model */
} geniex_EmbedderCreateInput;

/**
 * @brief Create and initialize an embedder instance from model files
 *
 * @param input[in]: Input parameters for the embedder creation
 * @param out_handle[out]: Pointer to the embedder handle. Must be freed with geniex_embedder_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_embedder_create(const geniex_EmbedderCreateInput* input, geniex_Embedder** out_handle);

/**
 * @brief Destroy embedder instance and free associated resources
 *
 * @param handle[in]: The embedder handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_embedder_destroy(geniex_Embedder* handle);

/* ====================  Embedding Generation  ============================= */

/** Input structure for embedding generation */
typedef struct {
    const char**                  texts;        /** Array of input texts in UTF-8 encoding */
    int32_t                       text_count;   /** Number of input texts */
    const geniex_EmbeddingConfig* config;       /** Embedding configuration (optional, can be nullptr) */
    const int32_t**               input_ids_2d; /** 2D array of already tokenized raw input ids.
                                                 * When passed in, texts will be ignored.
                                                 * NOTE: this is supported for llama_cpp backend only.
                                                 * Passing this param to other backends will be ignored */
    const int32_t* input_ids_row_lengths;       /** Array containing the length of each row in input_ids_2d */
    int32_t        input_ids_row_count;         /** Number of rows in input_ids_2d array */
    const char*    task_type;                   /** Task type: "query", "document" */

    /* ====================  Image inputs (for multimodal embedders)  ============================= */

    geniex_Path* image_paths; /** Array of image file paths to embed (UTF-8).
                               *  When non-NULL and image_count > 0, the call is treated as
                               *  an image embedding request.
                               *
                               *  Each path should point to an image file in a supported format
                               *  for the underlying plugin/model (e.g. PNG/JPEG).
                               *
                               *  NOTE: Text/token inputs and image inputs are mutually exclusive.
                               *  Providing both will result in GENIEX_ERROR_COMMON_INVALID_INPUT. */
    int32_t image_count;      /** Number of images in image_paths. */

    /* ====================  Video inputs (for multimodal embedders)  ============================= */

    geniex_Path* video_paths; /** Array of video file paths to embed (UTF-8).
                               *  When non-NULL and video_count > 0, the call is treated as
                               *  a video embedding request.
                               *
                               *  Each path should point to a video file in a supported format
                               *  for the underlying plugin/model (e.g. MP4/AVI). */
    float* video_starts;      /** Array of start times for each video in seconds. Should be the same length as
                                      video_count.*/
    float* video_ends;   /** Array of end times for each video in seconds. Should be the same length as video_count.
                          * NOTE: video_starts and video_ends must be provided together. */
    int32_t video_count; /** Number of videos in video_paths, video_starts, and video_ends. */
} geniex_EmbedderEmbedInput;

/** Output structure for embedding generation */
typedef struct {
    float*             embeddings;      /** Output embeddings array (caller must free with geniex_free) */
    int32_t            embedding_count; /** Number of embeddings returned */
    geniex_ProfileData profile_data;    /** Profiling data for the embedding generation */
} geniex_EmbedderEmbedOutput;

/**
 * @brief Generate embeddings for input texts
 *
 * @param handle[in]: Embedder handle
 * @param input[in]: Input parameters for embedding generation
 * @param output[out]: Output data containing the generated embeddings
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_embedder_embed(
    geniex_Embedder* handle, const geniex_EmbedderEmbedInput* input, geniex_EmbedderEmbedOutput* output);

/* ====================  Model Information  ================================ */

/** Output structure for getting embedding dimension */
typedef struct {
    int32_t dimension; /** The embedding dimension size */
} geniex_EmbedderDimOutput;

/**
 * @brief Get embedding dimension from the model
 *
 * @param handle[in]: Embedder handle
 * @param output[out]: Output data containing the embedding dimension
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe
 */
GENIEX_API int32_t geniex_embedder_embedding_dim(const geniex_Embedder* handle, geniex_EmbedderDimOutput* output);

/* ========================================================================== */
/*                              RERANKING MODELS                               */
/* ========================================================================== */

/** Reranking configuration */
typedef struct {
    int32_t     batch_size;       /* Processing batch size */
    bool        normalize;        /* Whether to normalize scores */
    const char* normalize_method; /* Normalization: "softmax", "min-max", "none" */
} geniex_RerankConfig;

typedef struct geniex_Reranker geniex_Reranker; /* Opaque reranker handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating a reranker */
typedef struct {
    const char*        model_name;     /** Name of the model */
    geniex_Path        model_path;     /** Path to the model file */
    geniex_Path        tokenizer_path; /** Path to the tokenizer file */
    geniex_ModelConfig config;         /** Model configuration */
    geniex_PluginId    plugin_id;      /** Plugin to use for the model */
    const char*        device_id;      /** device to use for the model */
} geniex_RerankerCreateInput;

/**
 * @brief Create and initialize a reranker instance from model files
 *
 * @param input[in]: Input parameters for the reranker creation
 * @param out_handle[out]: Pointer to the reranker handle. Must be freed with geniex_reranker_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_reranker_create(const geniex_RerankerCreateInput* input, geniex_Reranker** out_handle);

/**
 * @brief Destroy reranker instance and free associated resources
 *
 * @param handle[in]: The reranker handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_reranker_destroy(geniex_Reranker* handle);

/* ====================  Reranking  ========================================= */

/** Input structure for reranking operation */
typedef struct {
    const char*                query;           /** Query text in UTF-8 encoding */
    const char**               documents;       /** Array of document texts in UTF-8 encoding */
    int32_t                    documents_count; /** Number of documents */
    const geniex_RerankConfig* config;          /** Reranking configuration (optional, can be nullptr) */
} geniex_RerankerRerankInput;

/** Output structure for reranking operation */
typedef struct {
    float*             scores;       /** Output ranking scores array (caller must free with geniex_free) */
    int32_t            score_count;  /** Number of scores returned */
    geniex_ProfileData profile_data; /** Profiling data for the reranking operation */
} geniex_RerankerRerankOutput;

/**
 * @brief Rerank documents against a query
 *
 * @param handle[in]: Reranker handle
 * @param input[in]: Input parameters for reranking operation
 * @param output[out]: Output data containing the ranking scores
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_reranker_rerank(
    geniex_Reranker* handle, const geniex_RerankerRerankInput* input, geniex_RerankerRerankOutput* output);

/* ========================================================================== */
/*                              IMAGE GENERATION                               */
/* ========================================================================== */

/* ====================  Configuration Structures  =========================== */

/** Image generation sampling parameters */
typedef struct {
    const char* method;         /* Sampling method: "ddim", "ddpm", etc. */
    int32_t     steps;          /* Number of denoising steps */
    float       guidance_scale; /* Classifier-free guidance scale */
    float       eta;            /* DDIM eta parameter */
    int32_t     seed;           /* Random seed (-1 for random) */
} geniex_ImageSamplerConfig;

/** Diffusion scheduler configuration */
typedef struct {
    const char* type;                /* Scheduler type: "ddim", etc. */
    int32_t     num_train_timesteps; /* Training timesteps */
    int32_t     steps_offset;        /* An offset added to the inference steps */
    float       beta_start;          /* Beta schedule start */
    float       beta_end;            /* Beta schedule end */
    const char* beta_schedule;       /* Beta schedule: "scaled_linear" */
    const char* prediction_type;     /* Prediction type: "epsilon", "v_prediction" */
    const char* timestep_type;       /* Timestep type: "discrete", "continuous" */
    const char* timestep_spacing;    /* Timestep spacing: "linspace", "leading", "trailing" */
    const char* interpolation_type;  /* Interpolation type: "linear", "exponential" */
    geniex_Path config_path;         /* Optional config file path */
} geniex_SchedulerConfig;

/** Image generation configuration */
typedef struct {
    const char**              prompts;               /* Required positive prompts */
    int32_t                   prompt_count;          /* Number of positive prompts */
    const char**              negative_prompts;      /* Optional negative prompts */
    int32_t                   negative_prompt_count; /* Number of negative prompts */
    int32_t                   height;                /* Output image height */
    int32_t                   width;                 /* Output image width */
    geniex_ImageSamplerConfig sampler_config;        /* Sampling parameters */
    geniex_SchedulerConfig    scheduler_config;      /* Scheduler configuration */
    float                     strength;              /* Denoising strength for img2img */
} geniex_ImageGenerationConfig;

typedef struct geniex_ImageGen geniex_ImageGen; /* Opaque image generator handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating an image generator */
typedef struct {
    const char*        model_name;            /** Name of the model */
    geniex_Path        model_path;            /** Path to the model file */
    geniex_ModelConfig config;                /** Model configuration */
    geniex_Path        scheduler_config_path; /** Path to the scheduler config file */
    geniex_PluginId    plugin_id;             /** Plugin to use for the model */
    const char*        device_id;             /** Device to use for the model, NULL for default device */
} geniex_ImageGenCreateInput;

/**
 * @brief Create and initialize an image generator instance
 *
 * @param input[in]: Input parameters for the image generator creation
 * @param out_handle[out]: Pointer to the image generator handle. Must be freed with geniex_imagegen_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_imagegen_create(const geniex_ImageGenCreateInput* input, geniex_ImageGen** out_handle);

/**
 * @brief Destroy image generator instance and free associated resources
 *
 * @param handle[in]: The image generator handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_imagegen_destroy(geniex_ImageGen* handle);

/* ====================  Image Generation  ================================== */

/** Input structure for text-to-image generation */
typedef struct {
    const char*                         prompt_utf8; /** Text prompt in UTF-8 encoding */
    const geniex_ImageGenerationConfig* config;      /** Image generation configuration */
    geniex_Path                         output_path; /** Optional output file path (NULL for auto-generated) */
} geniex_ImageGenTxt2ImgInput;

/** Input structure for image-to-image generation */
typedef struct {
    geniex_Path                         init_image_path; /** Path to initial image file for img2img */
    const char*                         prompt_utf8;     /** Text prompt in UTF-8 encoding */
    const geniex_ImageGenerationConfig* config;          /** Image generation configuration */
    geniex_Path                         output_path;     /** Optional output file path (NULL for auto-generated) */
} geniex_ImageGenImg2ImgInput;

/** Output structure for image generation */
typedef struct {
    geniex_Path
        output_image_path; /** Path where the generated image will be saved (caller must free with geniex_free) */
} geniex_ImageGenOutput;

/**
 * @brief Generate image from text prompt and save to filesystem
 *
 * @param handle[in]: Image generator handle
 * @param input[in]: Input parameters for text-to-image generation
 * @param output[out]: Output data containing the path where the generated image is saved
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_imagegen_txt2img(
    geniex_ImageGen* handle, const geniex_ImageGenTxt2ImgInput* input, geniex_ImageGenOutput* output);

/**
 * @brief Generate image from initial image file and prompt, save to filesystem
 *
 * @param handle[in]: Image generator handle
 * @param input[in]: Input parameters for image-to-image generation (includes initial image path)
 * @param output[out]: Output data containing the path where the generated image is saved
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_imagegen_img2img(
    geniex_ImageGen* handle, const geniex_ImageGenImg2ImgInput* input, geniex_ImageGenOutput* output);

/* ========================================================================== */
/*                              SPEECH RECOGNITION (ASR)                       */
/* ========================================================================== */

/* ====================  Configuration Structures  =========================== */

/** ASR processing configuration */
typedef struct {
    const char* timestamps; /* Timestamp mode: "none", "segment", "word" */
    int32_t     beam_size;  /* Beam search size */
    bool        stream;     /* Enable streaming mode */
} geniex_ASRConfig;

/** ASR transcription result */
typedef struct {
    char*   transcript;        /* Transcribed text (UTF-8, caller must free with geniex_free) */
    float*  confidence_scores; /* Confidence scores for each unit (caller must free with geniex_free) */
    int32_t confidence_count;  /* Number of confidence scores */
    float*  timestamps;        /* Timestamp pairs: [start, end] for each unit (caller must free with geniex_free) */
    int32_t timestamp_count;   /* Number of timestamp pairs */
} geniex_ASRResult;

typedef struct geniex_ASR geniex_ASR; /* Opaque ASR handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating an ASR instance */
typedef struct {
    const char*        model_name;     /** Name of the model */
    geniex_Path        model_path;     /** Path to the model file */
    geniex_Path        tokenizer_path; /** Path to the tokenizer file (may be NULL) */
    geniex_ModelConfig config;         /** Model configuration */
    const char*        language;       /** Language code (ISO 639-1 or NULL) */
    geniex_PluginId    plugin_id;      /** Plugin to use for the model */
    const char*        device_id;      /** Device to use for the model, NULL for default device */
    const char* license_id;  /** licence id for loading NPU models, must be provided upon the first use of the license
                            key. null terminated string */
    const char* license_key; /** licence key for loading NPU models, null terminated string */
} geniex_AsrCreateInput;

/**
 * @brief Create and initialize an ASR instance with language support
 *
 * @param input[in]: Input parameters for the ASR creation
 * @param out_handle[out]: Pointer to the ASR handle. Must be freed with geniex_asr_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_create(const geniex_AsrCreateInput* input, geniex_ASR** out_handle);

/**
 * @brief Destroy ASR instance and free associated resources
 *
 * @param handle[in]: The ASR handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_destroy(geniex_ASR* handle);

/* ====================  Transcription  ===================================== */

/** Input structure for ASR transcription */
typedef struct {
    geniex_Path             audio_path; /** Path to audio file */
    const char*             language;   /** Language code (ISO 639-1 or NULL for auto-detect) */
    const geniex_ASRConfig* config;     /** ASR configuration (optional, can be nullptr) */
} geniex_AsrTranscribeInput;

/** Output structure for ASR transcription */
typedef struct {
    geniex_ASRResult   result;       /** Transcription result (caller must free with geniex_free for text fields) */
    geniex_ProfileData profile_data; /** Profiling data for the transcription operation */
} geniex_AsrTranscribeOutput;

/**
 * @brief Transcribe audio file to text with specified language
 *
 * @param handle[in]: ASR handle
 * @param input[in]: Input parameters for transcription (includes audio file path and language)
 * @param output[out]: Output data containing the transcription result
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_transcribe(
    geniex_ASR* handle, const geniex_AsrTranscribeInput* input, geniex_AsrTranscribeOutput* output);

/* ====================  Language Management  ============================== */

/** Input structure for getting supported languages */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} geniex_AsrListSupportedLanguagesInput;

/** Output structure for getting supported languages */
typedef struct {
    const char** language_codes; /** Array of supported language codes (caller must free with geniex_free) */
    int32_t      language_count; /** Number of supported languages */
} geniex_AsrListSupportedLanguagesOutput;

/**
 * @brief Get list of supported languages for ASR model
 *
 * @param handle[in]: ASR handle
 * @param input[in]: Input parameters for language list query
 * @param output[out]: Output data containing the supported languages array and count
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe
 */
GENIEX_API int32_t geniex_asr_list_supported_languages(const geniex_ASR* handle,
    const geniex_AsrListSupportedLanguagesInput* input, geniex_AsrListSupportedLanguagesOutput* output);

/* ========================================================================== */
/*                              ASR STREAMING                                  */
/* ========================================================================== */

/* ====================  Streaming Types & Callbacks  ====================== */

/** Callback for streaming transcription updates */
typedef void (*geniex_asr_transcription_callback)(const char* text, void* user_data);

/** ASR streaming configuration */
typedef struct {
    float       chunk_duration;   /* Duration in seconds for each chunk (default: 4.0) */
    float       overlap_duration; /* Overlap between chunks in seconds (default: 3.0) */
    int32_t     sample_rate;      /* Audio sample rate (default: 16000) */
    int32_t     max_queue_size;   /* Maximum chunks in processing queue (default: 10) */
    int32_t     buffer_size;      /* Audio buffer size for input (default: 512) */
    const char* timestamps;       /* Timestamp mode: "none", "segment", "word" */
    int32_t     beam_size;        /* Beam search size */
} geniex_ASRStreamConfig;

/* ====================  Streaming Operations  ============================== */

/** Input structure for beginning ASR streaming */
typedef struct {
    const geniex_ASRStreamConfig*     stream_config;    /** Streaming configuration (optional) */
    const char*                       language;         /** Language code (optional) */
    geniex_asr_transcription_callback on_transcription; /** Required: transcription updates */
    void*                             user_data;        /** User data passed to callbacks */
} geniex_AsrStreamBeginInput;

/** Output structure for streaming begin (minimal) */
typedef struct {
    void* reserved; /** Reserved for future use */
} geniex_AsrStreamBeginOutput;

/**
 * @brief Begin streaming ASR with specified callbacks
 *
 * @param handle[in]: ASR handle
 * @param input[in]: Streaming callbacks configuration
 * @param output[out]: Output structure (reserved for future use)
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_stream_begin(
    geniex_ASR* handle, const geniex_AsrStreamBeginInput* input, geniex_AsrStreamBeginOutput* output);

/** Input structure for processing audio data */
typedef struct {
    const float* audio_data; /** Audio samples (float32) */
    int32_t      length;     /** Number of samples */
} geniex_AsrStreamPushAudioInput;

/**
 * @brief Push audio data to streaming ASR for processing
 *
 * @param handle[in]: ASR handle
 * @param input[in]: Audio data to process
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_stream_push_audio(geniex_ASR* handle, const geniex_AsrStreamPushAudioInput* input);

/** Input structure for stopping streaming */
typedef struct {
    bool graceful; /** If true, processes remaining audio before stopping; if false, stops immediately */
} geniex_AsrStreamStopInput;

/**
 * @brief Stop streaming ASR
 *
 * @param handle[in]: ASR handle
 * @param input[in]: Stop configuration (graceful vs immediate)
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_asr_stream_stop(geniex_ASR* handle, const geniex_AsrStreamStopInput* input);

/* ========================================================================== */
/*                              TEXT-TO-SPEECH (TTS)                         */
/* ========================================================================== */

/* ====================  Configuration Structures  =========================== */

/** TTS synthesis configuration */
typedef struct {
    const char* voice;       /* Voice identifier */
    float       speed;       /* Speech speed (1.0 = normal) */
    int32_t     seed;        /* Random seed (-1 for random) */
    int32_t     sample_rate; /* Output sample rate in Hz */
} geniex_TTSConfig;

/** TTS sampling parameters */
typedef struct {
    float temperature;  /* Sampling temperature */
    float noise_scale;  /* Noise scale for voice variation */
    float length_scale; /* Length scale for speech duration */
} geniex_TTSSamplerConfig;

/** TTS synthesis result */
typedef struct {
    geniex_Path audio_path;       /* Path where the synthesized audio is saved (caller must free with geniex_free) */
    float       duration_seconds; /* Audio duration in seconds */
    int32_t     sample_rate;      /* Audio sample rate in Hz */
    int32_t     channels;         /* Number of audio channels (default: 1) */
    int32_t     num_samples;      /* Number of audio samples */
} geniex_TTSResult;

typedef struct geniex_TTS geniex_TTS; /* Opaque TTS handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating a TTS instance */
typedef struct {
    const char*        model_name;   /** Name of the model */
    geniex_Path        model_path;   /** Path to the TTS model file */
    geniex_ModelConfig config;       /** Model configuration */
    geniex_Path        vocoder_path; /** Path to the vocoder file */
    geniex_PluginId    plugin_id;    /** Plugin to use for the model */
    const char*        device_id;    /** Device to use for the model, NULL for default device */
} geniex_TtsCreateInput;

/**
 * @brief Create and initialize a TTS instance with model and vocoder
 *
 * @param input[in]: Input parameters for the TTS creation
 * @param out_handle[out]: Pointer to the TTS handle. Must be freed with geniex_tts_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_tts_create(const geniex_TtsCreateInput* input, geniex_TTS** out_handle);

/**
 * @brief Destroy TTS instance and free associated resources
 *
 * @param handle[in]: The TTS handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_tts_destroy(geniex_TTS* handle);

/* ====================  Speech Synthesis  ================================== */

/** Input structure for TTS synthesis */
typedef struct {
    const char*             text_utf8;   /** Text to synthesize in UTF-8 encoding */
    const geniex_TTSConfig* config;      /** TTS configuration (optional, can be nullptr) */
    geniex_Path             output_path; /** Optional output file path (NULL for auto-generated) */
} geniex_TtsSynthesizeInput;

/** Output structure for TTS synthesis */
typedef struct {
    geniex_TTSResult   result;       /** Synthesis result with audio saved to filesystem */
    geniex_ProfileData profile_data; /** Profiling data for the synthesis operation */
} geniex_TtsSynthesizeOutput;

/**
 * @brief Synthesize speech from text and save to filesystem
 *
 * @param handle[in]: TTS handle
 * @param input[in]: Input parameters for speech synthesis
 * @param output[out]: Output data containing the path where synthesized audio is saved
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_tts_synthesize(
    geniex_TTS* handle, const geniex_TtsSynthesizeInput* input, geniex_TtsSynthesizeOutput* output);

/* ====================  Voice Management  ================================== */

/** Input structure for getting available voices */
typedef struct {
    void* reserved; /** Reserved for future use, safe to set as NULL */
} geniex_TtsListAvailableVoicesInput;

/** Output structure for getting available voices */
typedef struct {
    const char** voice_ids;   /** Array of available voice identifiers (caller must free with geniex_free) */
    int32_t      voice_count; /** Number of available voices */
} geniex_TtsListAvailableVoicesOutput;

/**
 * @brief Get list of available voice identifiers
 *
 * @param handle[in]: TTS handle
 * @param input[in]: Input parameters for voice list query
 * @param output[out]: Output data containing the available voices array and count
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Thread-safe
 */
GENIEX_API int32_t geniex_tts_list_available_voices(const geniex_TTS* handle,
    const geniex_TtsListAvailableVoicesInput* input, geniex_TtsListAvailableVoicesOutput* output);

/* ========================================================================== */
/*                              COMPUTER VISION (CV)                           */
/* ========================================================================== */

/* ====================  Generic CV Data Types  ============================= */
/** Generic bounding box structure */
typedef struct {
    float x;      /* X coordinate (normalized or pixel, depends on model) */
    float y;      /* Y coordinate (normalized or pixel, depends on model) */
    float width;  /* Width */
    float height; /* Height */
} geniex_BoundingBox;

/** Generic detection/classification result */
typedef struct {
    geniex_Path*       image_paths; /* Output image paths (caller must free with geniex_free) */
    int32_t            image_count; /* Number of output images */
    int32_t            class_id;    /* Class ID (example: ConvNext) */
    float              confidence;  /* Confidence score [0.0-1.0] */
    geniex_BoundingBox bbox;        /* Bounding box (example: YOLO) */
    const char*        text;        /* Text result (example: OCR) (caller must free with geniex_free) */
    float*             embedding; /* Feature embedding (example: CLIP embedding) (caller must free with geniex_free) */
    int32_t            embedding_dim; /* Embedding dimension */
    float*             mask;          /* Mask (example: segmentation mask) (caller must free with geniex_free) */
    int32_t            mask_h;        /* Mask height */
    int32_t            mask_w;        /* Mask width */
} geniex_CVResult;

/** CV capabilities */
typedef enum {
    GENIEX_CV_OCR            = 0, /* OCR */
    GENIEX_CV_CLASSIFICATION = 1, /* Classification */
    GENIEX_CV_SEGMENTATION   = 2, /* Segmentation */
    GENIEX_CV_CUSTOM         = 3, /* Custom task */
} geniex_CVCapabilities;

/** CV model preprocessing configuration */
typedef struct {
    geniex_CVCapabilities capabilities;   /* Capabilities */
    geniex_Path           det_model_path; /* detection model path */
    geniex_Path           rec_model_path; /* recognition model path */
    geniex_Path           char_dict_path; /* Character dictionary path */
} geniex_CVModelConfig;

/* ====================  Generic CV Model  ================================== */

typedef struct geniex_CV geniex_CV; /* Opaque CV model handle */

typedef struct {
    const char*          model_name; /** Name of the model */
    geniex_CVModelConfig config;     /** CV model configuration */
    geniex_PluginId      plugin_id;  /** Plugin to use for the model */
    const char*          device_id;  /** device to use for the model */
    const char* license_id;  /** licence id for loading NPU models, must be provided upon the first use of the license
                                key. null terminated string */
    const char* license_key; /** licence key for loading NPU models, null terminated string */
} geniex_CVCreateInput;

/* ====================  Lifecycle Management  ============================== */

/**
 * @brief Create and initialize a CV model
 *
 * @param input[in]: Input parameters for the CV model creation
 * @param out_handle[out]: Pointer to the CV model handle. Must be freed with geniex_cv_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_cv_create(const geniex_CVCreateInput* input, geniex_CV** out_handle);

/**
 * @brief Destroy CV model instance and free associated resources
 *
 * @param handle[in]: The CV model handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_cv_destroy(geniex_CV* handle);

/* ====================  Generic Inference  ================================= */
/** Input structure for CV inference */
typedef struct {
    const char* input_image_path; /* Input image path */
} geniex_CVInferInput;

/** Output structure for CV inference */
typedef struct {
    geniex_CVResult* results;      /* Array of CV results (caller must free with geniex_free) */
    int32_t          result_count; /* Number of CV results */
} geniex_CVInferOutput;

/**
 * @brief Perform inference on a single image
 *
 * @param handle[in]: The CV model handle
 * @param input[in]: Input parameters for the inference
 * @param output[out]: Output data containing the inference results
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 */
GENIEX_API int32_t geniex_cv_infer(
    const geniex_CV* handle, const geniex_CVInferInput* input, geniex_CVInferOutput* output);

/* ========================================================================== */
/*                              SPEAKER DIARIZATION                            */
/* ========================================================================== */

/* ====================  Configuration Structures  =========================== */

/** Diarization processing configuration */
typedef struct {
    int32_t min_speakers; /* Minimum number of speakers (0 = auto-detect) */
    int32_t max_speakers; /* Maximum number of speakers (0 = no limit) */
} geniex_DiarizeConfig;

/** Speech segment structure */
typedef struct {
    float start_time;    /* Segment start time in seconds */
    float end_time;      /* Segment end time in seconds */
    char* speaker_label; /* Speaker label (e.g., "SPEAKER_00") (caller must free with geniex_free) */
} geniex_DiarizeSpeechSegment;

typedef struct geniex_Diarize geniex_Diarize; /* Opaque diarization handle */

/* ====================  Lifecycle Management  ============================== */

/** Input structure for creating a diarization instance */
typedef struct {
    const char*        model_name; /** Name of the model */
    geniex_Path        model_path; /** Path to the model folder */
    geniex_ModelConfig config;     /** Model configuration */
    geniex_PluginId    plugin_id;  /** Plugin to use for the model */
    const char*        device_id;  /** Device to use for the model, NULL for default device */
    const char* license_id;  /** Licence id for loading NPU models, must be provided upon the first use of the license
                                key. null terminated string */
    const char* license_key; /** Licence key for loading NPU models, null terminated string */
} geniex_DiarizeCreateInput;

/**
 * @brief Create and initialize a diarization instance
 *
 * @param input[in]: Input parameters for the diarization instance creation
 * @param out_handle[out]: Pointer to the diarization handle. Must be freed with geniex_diarize_destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_diarize_create(const geniex_DiarizeCreateInput* input, geniex_Diarize** out_handle);

/**
 * @brief Destroy diarization instance and free associated resources
 *
 * @param handle[in]: The diarization handle to destroy.
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_diarize_destroy(geniex_Diarize* handle);

/* ====================  Diarization Inference  ================================ */

/** Input structure for diarization inference */
typedef struct {
    geniex_Path                 audio_path; /** Path to audio file */
    const geniex_DiarizeConfig* config;     /** Diarization configuration (optional, can be nullptr) */
} geniex_DiarizeInferInput;

/** Output structure for diarization inference */
typedef struct {
    geniex_DiarizeSpeechSegment* segments;      /** Array of speech segments (caller must free with geniex_free) */
    int32_t                      segment_count; /** Number of segments */
    int32_t                      num_speakers;  /** Total unique speakers detected */
    float                        duration;      /** Total audio duration in seconds */
    geniex_ProfileData           profile_data;  /** Profiling data for the diarization operation */
} geniex_DiarizeInferOutput;

/**
 * @brief Perform speaker diarization on audio file
 *
 * Determines "who spoke when" in the audio recording, producing time-stamped segments
 * with speaker labels. Segments are time-ordered and non-overlapping.
 *
 * @param handle[in]: Diarization handle
 * @param input[in]: Input parameters for diarization (audio file path and optional configuration)
 * @param output[out]: Output data containing the diarization results
 *
 * @return geniex_ErrorCode: GENIEX_SUCCESS on success, negative on failure.
 *
 * @thread_safety: Not thread-safe
 */
GENIEX_API int32_t geniex_diarize_infer(
    geniex_Diarize* handle, const geniex_DiarizeInferInput* input, geniex_DiarizeInferOutput* output);

#ifdef __cplusplus
} /* extern "C" */
#endif
