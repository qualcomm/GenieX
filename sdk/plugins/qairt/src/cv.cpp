#include "cv.h"

#include <memory>
#include <string>

#include "common.h"
#include "logging.h"
#include "ml.h"

namespace geniex {

QnnCv::QnnCv(std::string lib_path) : m_model_impl(nullptr), m_lib_path(std::move(lib_path)) {}

QnnCv::~QnnCv() {}

/**
 * @brief Create CV model implementation
 *
 * Android: paddleocr, yolov12, yolov13, rfdetr, rmbg-v2, table-transformer, depth-anything-v2.
 * Windows: yolov12, yolov13, paddleocr, convnext-tiny, rfdetr, rmbg-v2, realesrgan, table-transformer,
 * depth-anything-v2.
 * Linux: yolov12, convnext.
 *
 * @param input Model creation input with model name and config
 * @return ML_SUCCESS or error code
 */
int32_t QnnCv::create_impl(const ml_CVCreateInput* input) {
    if (!input || !input->model_name) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    GENIEX_LOG_TRACE("Creating specific npu CV model for: {}", fmt::to_string(input->model_name));

    auto model_name = std::string_view(input->model_name);

#if defined(__ANDROID__)
    // Android: paddleocr, yolov12, yolov13, rfdetr, rmbg-v2, table-transformer, depth-anything-v2 are supported
    if (model_name == "paddleocr") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_paddleocr());
    } else if (model_name == "yolov12" || model_name == "yolov13" || model_name == "yolov26") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_yolov12());
    } else if (model_name == "rfdetr") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_rfdetr());
    } else if (model_name == "rmbg-v2") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_rmbg_v2());
    } else if (model_name == "table-transformer") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_table_transformer());
    } else if (model_name == "depth-anything-v2") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_depth_anything());
    } else {
        GENIEX_LOG_ERROR("Unsupported CV model for Android: {}", input->model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

#elif defined(_WIN32)
    // Windows: all CV models supported
    if (model_name == "yolov12" || model_name == "yolov13" || model_name == "yolov26") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_yolov12());
    } else if (model_name == "paddleocr") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_paddleocr());
    } else if (model_name == "convnext-tiny") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_convnext());
    } else if (model_name == "rfdetr") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_rfdetr());
    } else if (model_name == "rmbg-v2") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_rmbg_v2());
    } else if (model_name == "realesrgan") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_realesrgan());
    } else if (model_name == "table-transformer") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_table_transformer());
    } else if (model_name == "depth-anything-v2") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_depth_anything());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for npu CV implementation: {}", input->model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

#elif defined(__linux__)
    // Linux: convnext and yolov12 are supported
    if (model_name == "convnext") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_convnext());
    } else if (model_name == "yolov12" || model_name == "yolov13" || model_name == "yolov26") {
        m_model_impl = std::unique_ptr<ICv>(create_qnn_yolov12());
    } else {
        GENIEX_LOG_ERROR("Unsupported model name for Linux npu CV implementation: {}", input->model_name);
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

#endif

    if (!m_model_impl) {
        GENIEX_LOG_ERROR("Failed to create specific model implementation");
        return ML_ERROR_COMMON_MODEL_LOAD;
    }

    // inject qnn path, use class for c_str lifetime management
    QnnFolderPathFiller _(input, m_lib_path);
    return m_model_impl->create(input);
}

/**
 * @brief Run CV model inference on image
 *
 * @param input Image path and inference config
 * @param output Model-specific detection/classification results
 * @return ML_SUCCESS or error code
 */
int32_t QnnCv::infer(const ml_CVInferInput* input, ml_CVInferOutput* output) {
    if (!m_model_impl || !output) {
        return ML_ERROR_COMMON_INVALID_INPUT;
    }

    return m_model_impl->infer(input, output);
}

}  // namespace geniex
