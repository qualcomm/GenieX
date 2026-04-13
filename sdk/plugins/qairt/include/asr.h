#pragma once

#include <memory>
#include <string>

#include "ml.h"
#include "plugin/IAsr.h"

namespace geniex {

/**
 * @brief QNN-accelerated Automatic Speech Recognition implementation
 *
 * Supports both batch transcription and streaming ASR on Qualcomm NPU.
 */
class QnnAsr : public IAsr {
   public:
    explicit QnnAsr(std::string lib_path = "");
    virtual ~QnnAsr() override;

    // Regular ASR interface
    virtual int32_t transcribe(const ml_AsrTranscribeInput*, ml_AsrTranscribeOutput*) override;
    virtual int32_t list_supported_languages(
        const ml_AsrListSupportedLanguagesInput*, ml_AsrListSupportedLanguagesOutput*) override;

    // Streaming ASR interface
    virtual int32_t stream_begin(const ml_AsrStreamBeginInput* input, ml_AsrStreamBeginOutput* output) override;
    virtual int32_t stream_push_audio(const ml_AsrStreamPushAudioInput* input) override;
    virtual int32_t stream_stop(const ml_AsrStreamStopInput* input) override;

   protected:
    virtual int32_t create_impl(const ml_AsrCreateInput* input) override;

   private:
    std::unique_ptr<IAsr> m_model_impl;
    std::string           m_lib_path;
};

}  // namespace geniex
   //
// export from qnn-run
extern "C" {
#if defined(__ANDROID__)
geniex::IAsr* create_qnn_parakeet();
geniex::IAsr* create_qnn_wav2vec2();

#elif defined(_WIN32)
geniex::IAsr* create_qnn_parakeet();
geniex::IAsr* create_qnn_wav2vec2();

#elif defined(__linux__)
geniex::IAsr* create_qnn_parakeet();

#endif
}
