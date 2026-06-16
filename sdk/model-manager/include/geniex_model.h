#pragma once

/**
 * @file geniex_model.h
 * @brief C API for model management: download, local storage, and path resolution.
 *
 * Implemented in Rust (sdk/model-manager), compiled to libgeniex_model.a and
 * linked into libgeniex.so via the GENIEX_MODEL_MANAGER CMake option.
 *
 * Memory convention (mirrors geniex.h):
 *   - All output char* fields are heap-allocated; free them with geniex_free().
 *   - Dedicated _free helpers (geniex_model_paths_free, geniex_model_list_free)
 *     free their members and zero the struct.
 *   - Input pointers are caller-owned and never freed by this library.
 *
 * Error codes reuse the geniex_ErrorCode range defined in geniex.h (negative = error).
 */

#include <stdbool.h>
#include <stdint.h>

#include "geniex.h" /* GENIEX_API, geniex_ErrorCode, geniex_Path, geniex_free() */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 *  Initialization
 * ============================================================ */

/**
 * @brief Initialize the model manager.
 *
 * Must be called before any other geniex_model_* function.
 *
 * `data_dir` precedence: argument → `GENIEX_DATADIR` env → `~/.cache/geniex`.
 *
 * Calling this function more than once per process is a programmer error;
 * a warning is logged via geniex_set_log and
 * GENIEX_ERROR_COMMON_INVALID_INPUT is returned. The first successful
 * call is authoritative.
 *
 * HuggingFace tokens are a per-pull concern — see
 * `geniex_ModelPullInput.hf_token` — and are deliberately NOT accepted
 * here, so that callers can rotate credentials between downloads.
 *
 * @param data_dir  Local cache directory, or NULL to fall back to env/default.
 * @return GENIEX_SUCCESS on success, negative geniex_ErrorCode on failure.
 */
GENIEX_API int32_t geniex_model_init(geniex_Path data_dir);

/**
 * @brief Deinitialize the model manager and release resources.
 * @return GENIEX_SUCCESS.
 */
GENIEX_API int32_t geniex_model_deinit(void);

/**
 * @brief Human-readable message for the most recent failure on the calling
 *        thread.
 *
 * Every geniex_model_* function that returns a negative error code also
 * records a detailed message (e.g. "quantization 'Q2_K' not found for model
 * 'org/repo'") that the int code alone can't convey. The message is
 * thread-local and overwritten by the next failing call on that thread.
 *
 * @return A NUL-terminated, library-owned string (do NOT free) valid until the
 *         next geniex_model_* call on this thread, or NULL if no error has been
 *         recorded yet.
 */
GENIEX_API const char* geniex_model_last_error_message(void);

/* ============================================================
 *  Model type
 * ============================================================ */

typedef enum {
    GENIEX_MODEL_TYPE_LLM = 0,
    GENIEX_MODEL_TYPE_VLM = 1,
} geniex_ModelType;

/* ============================================================
 *  Path resolution
 * ============================================================ */

/**
 * @brief Resolved absolute file paths for a loaded model.
 *
 * All non-NULL char* fields are heap-allocated.
 * Free the entire struct with geniex_model_paths_free().
 */
typedef struct {
    char*            model_path;     /**< Main model file (absolute path).          */
    char*            mmproj_path;    /**< Multimodal projection file. NULL if unused. */
    char*            tokenizer_path; /**< Tokenizer file. NULL if unused.            */
    char*            model_dir;      /**< Model directory (always set).              */
    char*            model_name;     /**< Architecture name, e.g. "qwen3-4b".       */
    char*            plugin_id;      /**< Plugin ID, e.g. "llama_cpp".               */
    geniex_ModelType model_type;     /**< LLM or VLM.                          */
} geniex_ModelPaths;

/**
 * @brief Get resolved file paths for a model.
 *
 * @param model_name  "org/repo" or "org/repo:quant".
 *                    If quant is omitted the first downloaded quantization is used.
 * @param out_paths   Populated on success. Call geniex_model_paths_free() when done.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_get_paths(const char* model_name, geniex_ModelPaths* out_paths);

/** Free all heap strings inside geniex_ModelPaths and zero the struct. */
GENIEX_API void geniex_model_paths_free(geniex_ModelPaths* paths);

/* ============================================================
 *  Local cache management
 * ============================================================ */

/**
 * @brief Detailed metadata for one cached model.
 *
 * All non-NULL char* fields (and the `precisions` array) are heap-allocated;
 * free the enclosing output with geniex_model_list_detailed_free().
 */
typedef struct {
    char*            name;       /**< "org/repo".                          */
    char*            model_name; /**< Architecture name, e.g. "qwen3-4b". */
    char*            plugin_id;  /**< Plugin ID, e.g. "llama_cpp".         */
    geniex_ModelType model_type;
    int64_t          total_size;      /**< Sum of downloaded file sizes (bytes). */
    char**           precisions;      /**< Downloaded quant names.          */
    int32_t          precision_count; /**< Length of `precisions`.          */
} geniex_ModelDetail;

typedef struct {
    geniex_ModelDetail* models;
    int32_t             count;
} geniex_ModelListDetailedOutput;

/**
 * @brief List all locally cached models with full per-model metadata.
 * @param output  Populated on success. Call geniex_model_list_detailed_free() when done.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_list_detailed(geniex_ModelListDetailedOutput* output);

/** Free every model's strings + precisions array, then zero the struct. */
GENIEX_API void geniex_model_list_detailed_free(geniex_ModelListDetailedOutput* output);

/**
 * @brief Delete a cached model from disk.
 * @param model_name  "org/repo" format.
 * @return GENIEX_SUCCESS, or GENIEX_ERROR_COMMON_FILE_NOT_FOUND if not cached.
 */
GENIEX_API int32_t geniex_model_remove(const char* model_name);

/**
 * @brief Delete all cached models.
 * @param removed_count  Set to the number of deleted models. May be NULL.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_clean(int32_t* removed_count);

/**
 * @brief Get the model type of a cached model.
 * @param model_name  "org/repo" format.
 * @param out_type    Set on success.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_get_type(const char* model_name, geniex_ModelType* out_type);

/**
 * @brief Override the model type stored in a cached model's manifest.
 * @param model_name  "org/repo" format.
 * @param type        New model type.
 * @return GENIEX_SUCCESS, or GENIEX_ERROR_COMMON_FILE_NOT_FOUND if not cached.
 */
GENIEX_API int32_t geniex_model_set_type(const char* model_name, geniex_ModelType type);

/* ============================================================
 *  Download
 * ============================================================ */

typedef enum {
    GENIEX_HUB_AUTO        = 0, /**< Automatic hub selection              */
    GENIEX_HUB_HUGGINGFACE = 1, /**< HuggingFace Hub                      */
    GENIEX_HUB_MODELSCOPE  = 2, /**< ModelScope (mainland China preferred) */
    GENIEX_HUB_AIHUB       = 3, /**< Qualcomm AI Hub qairt assets         */
    GENIEX_HUB_VOLCES      = 4, /**< Volces TOS (mainland China preferred) */
    /**
     * Local filesystem — not a real hub. The value 127 (0x7F) keeps it
     * well separated from real hub identifiers so future additions won't
     * collide, and signals "this isn't a remote source" at a glance.
     */
    GENIEX_HUB_LOCALFS = 127,
} geniex_HubSource;

/**
 * @brief Per-file download progress, as reported to the user callback.
 *
 * Lifetimes: `file_name` and the enclosing array are library-owned and
 * valid only for the duration of the callback. Callbacks must copy any
 * data they wish to retain.
 */
typedef struct {
    const char* file_name; /**< Relative path within the model directory. */
    int64_t     downloaded_bytes;
    int64_t     total_bytes; /**< -1 if unknown */
} geniex_FileProgress;

/**
 * @brief Progress callback invoked periodically during a pull.
 *
 * @param files       Array of per-file progress entries (one per file being fetched).
 * @param file_count  Length of `files`.
 * @param user_data   Caller-provided pointer (as passed to geniex_model_pull).
 * @return true to continue, false to cancel the download.
 */
typedef bool (*geniex_download_progress_cb)(const geniex_FileProgress* files, int32_t file_count, void* user_data);

typedef struct {
    /**
     * MUST be `sizeof(geniex_ModelPullInput)` when the struct is
     * constructed. Lets the library reject callers that were compiled
     * against an older header after a field was added.
     *
     * Rationale: C doesn't expose a stable way to version a struct,
     * and language bindings (Python ctypes, Go cgo, Android JNI) all
     * mirror this layout by hand. A non-zero size lets the SDK detect
     * a layout mismatch before dereferencing uninitialized fields.
     *
     * Conventional initialisation:
     *   geniex_ModelPullInput in = {0};
     *   in.struct_size = sizeof(in);
     *   in.model_name = "...";
     *   ...
     *
     * Returns GENIEX_ERROR_COMMON_INVALID_INPUT if struct_size is zero
     * or not a recognised version.
     */
    uint32_t         struct_size;
    const char*      model_name; /**< "org/repo" or short alias                    */
    const char*      quant;      /**< Quantization hint. NULL for auto-select      */
    geniex_HubSource hub;        /**< Use GENIEX_HUB_AUTO for automatic selection  */
    geniex_Path      local_path; /**< Required only when hub == GENIEX_HUB_LOCALFS */
    /**
     * HuggingFace bearer token for this pull. NULL falls back to the
     * `GENIEX_HFTOKEN` environment variable; if that is also unset the
     * download proceeds anonymously (subject to HF rate limits). Only
     * consulted when `hub == GENIEX_HUB_HUGGINGFACE`.
     */
    const char* hf_token;
    /**
     * Target chipset for AI Hub (qairt) pulls, e.g. "SM8650". Matched
     * against the name/aliases fields of platform.json. Only consulted
     * when `hub == GENIEX_HUB_AIHUB`.
     *
     * NULL or an empty string asks the SDK to auto-detect the host
     * chipset. Detection currently works on Windows-on-Snapdragon
     * (X Elite / X Plus / X2 Elite); on other hosts an auto-detect
     * request fails with GENIEX_ERROR_COMMON_INVALID_INPUT and the
     * caller must pass a chipset explicitly.
     */
    const char* chipset;
    /**
     * AI Hub model `display_name`. Used when `hub == GENIEX_HUB_AIHUB` or
     * `hub == GENIEX_HUB_AUTO` resolves to AI Hub. If NULL and `model_name`
     * is `qualcomm/<repo>` or `qai-hub-models/<repo>`, `<repo>` is used as
     * the display_name; otherwise the caller must set this. `model_name`
     * still names the on-disk directory ("org/repo" shape), matching the
     * Go CLI's storedName/displayName split.
     */
    const char*                 display_name;
    geniex_download_progress_cb on_progress; /**< NULL to suppress progress reporting           */
    void*                       user_data;   /**< Forwarded to on_progress                      */
    /**
     * Optional model-type override written into the manifest as the pull
     * publishes it, so a caller that already knows the type doesn't need a
     * separate geniex_model_set_type() round-trip. Use -1 to auto-detect
     * (the default); 0 = LLM, 1 = VLM (matching geniex_ModelType).
     */
    int32_t model_type;
} geniex_ModelPullInput;

/** Pass as geniex_ModelPullInput.model_type to keep auto-detection. */
#define GENIEX_MODEL_TYPE_AUTO (-1)

/**
 * @brief Download a model (blocking).
 *
 * Supports resume: partially downloaded files are continued from where they
 * left off.  On success the model is immediately available via geniex_model_get_paths().
 *
 * @param input  Pull parameters. Must not be NULL.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_pull(const geniex_ModelPullInput* input);

/* ============================================================
 *  Query (plan without downloading)
 * ============================================================ */

/**
 * @brief One quantization advertised by the source for a model.
 *
 * `quant` is heap-allocated; freed by geniex_model_query_free().
 */
typedef struct {
    char*   quant;      /**< Quantization name, e.g. "Q4_K_M".            */
    int64_t size;       /**< Size in bytes of the largest file for this quant. */
    bool    is_default; /**< True for the quant the SDK would auto-select. */
} geniex_QuantCandidate;

/**
 * @brief Result of geniex_model_query.
 *
 * All non-NULL char* fields and the `candidates` array are heap-allocated;
 * free with geniex_model_query_free().
 */
typedef struct {
    char*                  model_name; /**< Architecture name.            */
    char*                  plugin_id;  /**< Plugin ID, e.g. "llama_cpp".  */
    geniex_ModelType       model_type;
    geniex_QuantCandidate* candidates;
    int32_t                candidate_count;
} geniex_ModelQueryOutput;

/**
 * @brief Resolve a model's remote candidate quantizations without downloading.
 *
 * Plans the model against its source (HF / AI Hub / LocalFS) and returns the
 * advertised quants + sizes so callers can present a precision picker before
 * committing to geniex_model_pull.
 *
 * Reuses geniex_ModelPullInput; the quant, on_progress, user_data, and
 * model_type fields are ignored.
 *
 * @param input  Pull-shaped parameters. Must not be NULL.
 * @param out    Populated on success. Call geniex_model_query_free() when done.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_query(const geniex_ModelPullInput* input, geniex_ModelQueryOutput* out);

/** Free all heap members of geniex_ModelQueryOutput and zero the struct. */
GENIEX_API void geniex_model_query_free(geniex_ModelQueryOutput* out);

/* ============================================================
 *  Platform alias resolution
 * ============================================================ */

/**
 * @brief Resolve a short alias to the canonical "org/repo" name for the
 *        current OS and CPU architecture.
 *
 * Example: "qwen3"   → "NexaAI/Qwen3-4B-GGUF"   (x86-64)
 *          "qwen3vl" → "NexaAI/Qwen3-VL-4B-NPU"  (Windows arm64)
 *
 * @param alias         Short model name.
 * @param out_full_name Set to a heap-allocated string on success.
 *                      Free with geniex_free().
 * @return GENIEX_SUCCESS if resolved, GENIEX_ERROR_COMMON_INVALID_INPUT if unknown alias.
 */
GENIEX_API int32_t geniex_model_resolve_alias(const char* alias, char** out_full_name);

/* ============================================================
 *  Chipset
 * ============================================================ */

/**
 * @brief One chipset AI Hub publishes assets for.
 *
 * `name` and the `aliases` array are heap-allocated; free the enclosing
 * list with geniex_model_list_chipsets_free().
 */
typedef struct {
    char*  name;         /**< Reference device, e.g. "Snapdragon X Elite CRD". */
    char** aliases;      /**< Accepted aliases, incl. the canonical id
                          *   "qualcomm-snapdragon-x-elite" and "sm8650".   */
    int32_t alias_count; /**< Length of `aliases`.                          */
} geniex_ChipsetInfo;

typedef struct {
    geniex_ChipsetInfo* chipsets;
    int32_t             count;
} geniex_ChipsetList;

/**
 * @brief List every chipset supported by Qualcomm AI Hub, with aliases.
 *
 * Sourced from the remote `platform.json` (cached on disk for 24h), so the
 * first call may hit the network.
 *
 * @param out  Populated on success. Call geniex_model_list_chipsets_free() when done.
 * @return GENIEX_SUCCESS, or a negative geniex_ErrorCode.
 */
GENIEX_API int32_t geniex_model_list_chipsets(geniex_ChipsetList* out);

/** Free every chipset's name + aliases array, then zero the struct. */
GENIEX_API void geniex_model_list_chipsets_free(geniex_ChipsetList* out);

/**
 * @brief Detect the chipset of the current host via a local probe.
 *
 * Pure local detection — no network. Coverage: Windows-on-Snapdragon
 * (X Elite / X Plus / X2 Elite), Linux on Qualcomm Dragonwing boards
 * (QCS6490 / QCS9075), and Android on Snapdragon (`ro.soc.model`).
 *
 * The returned string is not normalised across platforms: Windows/Linux
 * yield an AI Hub canonical chipset id (e.g. "qualcomm-snapdragon-x-elite"),
 * while Android yields the raw SoC model (e.g. "SM8750"). Both are accepted
 * by geniex_model_pull's `chipset` field.
 *
 * @param out_chipset  Set to a heap-allocated string on success, or NULL when
 *                     the host cannot be probed on this platform. Free a
 *                     non-NULL value with geniex_free().
 * @return GENIEX_SUCCESS (even when *out_chipset is NULL), or a negative
 *         geniex_ErrorCode on a hard failure (e.g. NULL out_chipset).
 */
GENIEX_API int32_t geniex_model_detect_chipset(char** out_chipset);

#ifdef __cplusplus
} /* extern "C" */
#endif
