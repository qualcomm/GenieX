#pragma once

#include <memory>
#include <string>

#include "ml.h"
#include "plugin/ICv.h"

namespace geniex {

/**
 * @brief QNN-accelerated Computer Vision model implementation
 *
 * Factory class for CV models (OCR, object detection, classification) on Qualcomm NPU.
 */
class QnnCv : public ICv {
   public:
    explicit QnnCv(std::string lib_path = "");
    virtual ~QnnCv() override;

    virtual int32_t infer(const ml_CVInferInput* input, ml_CVInferOutput* output) override;

   protected:
    virtual int32_t create_impl(const ml_CVCreateInput* input) override;

   private:
    std::unique_ptr<ICv> m_model_impl;
    std::string          m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
// Android supports paddleocr, yolov12, yolov13, rfdetr, rmbg_v2, table-transformer, depth-anything-v2
ML_API geniex::ICv* create_qnn_paddleocr();
ML_API geniex::ICv* create_qnn_yolov12();
ML_API geniex::ICv* create_qnn_rfdetr();
ML_API geniex::ICv* create_qnn_rmbg_v2();
ML_API geniex::ICv* create_qnn_table_transformer();
ML_API geniex::ICv* create_qnn_depth_anything();

#elif defined(_WIN32)
// Windows supports all CV models
ML_API geniex::ICv* create_qnn_yolov12();
ML_API geniex::ICv* create_qnn_paddleocr();
ML_API geniex::ICv* create_qnn_convnext();
ML_API geniex::ICv* create_qnn_rfdetr();
ML_API geniex::ICv* create_qnn_rmbg_v2();
ML_API geniex::ICv* create_qnn_realesrgan();
ML_API geniex::ICv* create_qnn_table_transformer();
ML_API geniex::ICv* create_qnn_depth_anything();

#elif defined(__linux__)
ML_API geniex::ICv* create_qnn_convnext();
ML_API geniex::ICv* create_qnn_yolov12();

#endif
}
