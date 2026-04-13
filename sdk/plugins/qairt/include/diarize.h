#pragma once

#include <memory>
#include <string>

#include "ml.h"
#include "plugin/IDiarize.h"

namespace geniex {

/**
 * @brief QNN-accelerated speaker diarization implementation
 *
 * Factory class for creating diarization models on Qualcomm NPU (Windows only).
 */
class QnnDiarize : public IDiarize {
   public:
    explicit QnnDiarize(std::string lib_path = "");
    virtual ~QnnDiarize() override;

    virtual int32_t infer(const ml_DiarizeInferInput* input, ml_DiarizeInferOutput* output) override;

   protected:
    virtual int32_t create_impl(const ml_DiarizeCreateInput* input) override;

   private:
    std::unique_ptr<IDiarize> m_model_impl;
    std::string               m_lib_path;
};

}  // namespace geniex

// export from qnn-run
extern "C" {
#if defined(__ANDROID__)

#elif defined(_WIN32)
ML_API geniex::IDiarize* create_qnn_pyannote();

#elif defined(__linux__)

#endif
}
