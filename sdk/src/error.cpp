#include "geniex.h"

/**
 * @brief Get error message string for error code
 *
 * This function maps error codes to human-readable error messages.
 * Error codes follow hierarchical naming: GENIEX_ERROR_[CATEGORY]_[SUBCATEGORY]_[ERROR_TYPE]
 *
 * @param error_code Error code from geniex_ErrorCode enumeration
 * @return Error message string (const char*)
 */
const char* geniex_get_error_message(const geniex_ErrorCode error_code) {
    if (error_code > 0) {
        return "Success";  // Success case
    }

    switch (error_code) {
            /** ===== SUCCESS ===== */
        case GENIEX_SUCCESS:
            return "Success";

            /* ===== COMMON ERRORS (100xxx) ===== */
        case GENIEX_ERROR_COMMON_UNKNOWN:
            return "Unknown error";
        case GENIEX_ERROR_COMMON_INVALID_INPUT:
            return "Invalid input parameters or handle";
        case GENIEX_ERROR_COMMON_INVALID_DEVICE:
            return "Unknown device alias (expected one of: cpu, gpu, npu, hybrid)";
        case GENIEX_ERROR_COMMON_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case GENIEX_ERROR_COMMON_FILE_NOT_FOUND:
            return "File not found or inaccessible";
        case GENIEX_ERROR_COMMON_NOT_INITIALIZED:
            return "Library not initialized";
        case GENIEX_ERROR_COMMON_NOT_SUPPORTED:
            return "Operation not supported";
        case GENIEX_ERROR_COMMON_MODEL_LOAD:
            return "Model loading failed";
        case GENIEX_ERROR_COMMON_MODEL_INVALID:
            return "Invalid model format";
        case GENIEX_ERROR_COMMON_PLUGIN_LOAD:
            return "Plugin loading failed";
        case GENIEX_ERROR_COMMON_PLUGIN_INVALID:
            return "Invalid plugin";
        case GENIEX_ERROR_COMMON_LICENSE_INVALID:
            return "Invalid license";
        case GENIEX_ERROR_COMMON_LICENSE_EXPIRED:
            return "License expired";

            /* ===== LLM ERRORS (200xxx) ===== */
        case GENIEX_ERROR_LLM_TOKENIZATION_FAILED:
            return "Tokenization failed";
        case GENIEX_ERROR_LLM_TOKENIZATION_CONTEXT_LENGTH:
            return "Context length exceeded";
        case GENIEX_ERROR_LLM_GENERATION_FAILED:
            return "Text generation failed";
        case GENIEX_ERROR_LLM_GENERATION_PROMPT_TOO_LONG:
            return "Input prompt too long";

            /* ===== VLM ERRORS (201xxx) ===== */
        case GENIEX_ERROR_VLM_IMAGE_LOAD:
            return "Image loading failed";
        case GENIEX_ERROR_VLM_IMAGE_FORMAT:
            return "Unsupported image format";
        case GENIEX_ERROR_VLM_AUDIO_LOAD:
            return "Audio loading failed";
        case GENIEX_ERROR_VLM_AUDIO_FORMAT:
            return "Unsupported audio format";
        case GENIEX_ERROR_VLM_GENERATION_FAILED:
            return "Multimodal generation failed";

            /* ===== Embedding ERRORS (202xxx) ===== */
        case GENIEX_ERROR_EMBEDDING_GENERATION:
            return "Embedding generation failed";
        case GENIEX_ERROR_EMBEDDING_DIMENSION:
            return "Invalid embedding dimension";

            /* ===== Reranking ERRORS (203xxx) ===== */
        case GENIEX_ERROR_RERANK_FAILED:
            return "Reranking failed";
        case GENIEX_ERROR_RERANK_INPUT:
            return "Invalid reranking input";

            /* ===== Image Generation ERRORS (204xxx) ===== */
        case GENIEX_ERROR_IMAGEGEN_GENERATION:
            return "Image generation failed";
        case GENIEX_ERROR_IMAGEGEN_PROMPT:
            return "Invalid image prompt";
        case GENIEX_ERROR_IMAGEGEN_DIMENSION:
            return "Invalid image dimensions";

            /* ===== ASR ERRORS (205xxx) ===== */

        case GENIEX_ERROR_ASR_TRANSCRIPTION:
            return "ASR transcription failed";
        case GENIEX_ERROR_ASR_AUDIO_FORMAT:
            return "Unsupported ASR audio format";
        case GENIEX_ERROR_ASR_LANGUAGE:
            return "Unsupported ASR language";

            /* ===== TTS ERRORS (206xxx) ===== */
        case GENIEX_ERROR_TTS_SYNTHESIS:
            return "TTS synthesis failed";
        case GENIEX_ERROR_TTS_VOICE:
            return "TTS voice not found";
        case GENIEX_ERROR_TTS_AUDIO_FORMAT:
            return "TTS audio format error";

            /* ===== CV ERRORS (207xxx) ===== */
        case GENIEX_ERROR_CV_OCR_DETECTION:
            return "OCR text detection failed";
        case GENIEX_ERROR_CV_OCR_RECOGNITION:
            return "OCR text recognition failed";

            /* ===== Diarization ERRORS (208xxx) ===== */
        case GENIEX_ERROR_DIARIZE_AUDIO_LOAD:
            return "Audio loading failed";
        case GENIEX_ERROR_DIARIZE_SEGMENTATION:
            return "Segmentation model execution failed";
        case GENIEX_ERROR_DIARIZE_EMBEDDING:
            return "Embedding extraction failed";
        case GENIEX_ERROR_DIARIZE_CLUSTERING:
            return "Speaker clustering failed";

        default:
            return "Unknown error code";
    }
}
