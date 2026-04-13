#include "plugin/Plugin.h"

#include <exception>
#include <string>

// Common includes for all platforms
#include "asr.h"
#include "common.h"
#include "cv.h"
#include "embedding.h"
#include "llm.h"
#include "logging.h"
#include "plugin/IAsr.h"
#include "plugin/ICv.h"
#include "plugin/IEmbedding.h"
#include "plugin/ILlm.h"
#include "plugin/IReranker.h"
#include "plugin/IVlm.h"
#include "rerank.h"
#include "vlm.h"

// Diarization only supported on Windows
#if defined(_WIN32)
#include "diarize.h"
#include "plugin/IDiarize.h"
#endif

namespace geniex {

class QairtPlugin : public Plugin {
public:
  QairtPlugin() : m_lib_path(compute_lib_path()) {
    GENIEX_LOG_INFO("QAIRT plugin initialized with lib path: {}", m_lib_path);
  }

  ~QairtPlugin() override {}

#if defined(__ANDROID__)
  // Android: LLM, VLM, ASR, CV, Embedding, and Rerank supported
  ILlm *create_llm() override { return new geniex::QnnLlm(m_lib_path); }
  IVlm *create_vlm() override { return new geniex::QnnVlm(m_lib_path); }
  IAsr *create_asr() override { return new geniex::QnnAsr(m_lib_path); }
  ICv *create_cv() override { return new geniex::QnnCv(m_lib_path); }
  IEmbedding *create_embedding() override {
    return new geniex::QnnEmbedding(m_lib_path);
  }
  IReranker *create_reranker() override {
    return new geniex::QnnReranker(m_lib_path);
  }

#elif defined(_WIN32)
  // Windows: all features supported
  ICv *create_cv() override { return new geniex::QnnCv(m_lib_path); }
  ILlm *create_llm() override { return new geniex::QnnLlm(m_lib_path); }
  IAsr *create_asr() override { return new geniex::QnnAsr(m_lib_path); }
  IVlm *create_vlm() override { return new geniex::QnnVlm(m_lib_path); }
  IEmbedding *create_embedding() override {
    return new geniex::QnnEmbedding(m_lib_path);
  }
  IReranker *create_reranker() override {
    return new geniex::QnnReranker(m_lib_path);
  }
  IDiarize *create_diarize() override {
    return new geniex::QnnDiarize(m_lib_path);
  }

#elif defined(__linux__)
  ILlm *create_llm() override { return new geniex::QnnLlm(m_lib_path); }
  IVlm *create_vlm() override { return new geniex::QnnVlm(m_lib_path); }
  IAsr *create_asr() override { return new geniex::QnnAsr(m_lib_path); }
  ICv *create_cv() override { return new geniex::QnnCv(m_lib_path); }
  IEmbedding *create_embedding() override {
    return new geniex::QnnEmbedding(m_lib_path);
  }
  IReranker *create_reranker() override {
    return new geniex::QnnReranker(m_lib_path);
  }
#endif

private:
  std::string m_lib_path;

  /**
   * @brief Compute the QNN lib folder path once at plugin initialization.
   *
   * On Windows: detects HTP arch version and selects the appropriate directory
   * (htp-files-v81 if arch is v81 and the directory exists, otherwise
   * htp-files). On Android: uses "htp-files" as-is (no device suffix needed).
   * On Linux: returns "" so that each model resolves its path using the
   * per-model device_id at create time (required for device-specific toolchain
   * suffixes).
   */
  static std::string compute_lib_path() {
#if defined(_WIN32) || defined(__ANDROID__)
    int arch = detect_htp_arch();
    return fill_qnn_lib_path("", arch);
#else
    // Linux: lib path depends on per-model device_id; defer to model creation
    // time.
    return "";
#endif
  }
};

} // namespace geniex

/**
 * @brief Get QAIRT plugin identifier
 *
 * @return GENIEX_PLUGIN_ID_QAIRT
 */
ml_PluginId plugin_id() { return GENIEX_PLUGIN_ID_QAIRT; }

/**
 * @brief Create QAIRT plugin instance for Qualcomm NPU acceleration
 *
 * @return QairtPlugin instance or nullptr on failure
 */
geniex::Plugin *create_plugin() {
  GENIEX_LOG_TRACE("creating qairt plugin");

  try {
    return new geniex::QairtPlugin;
  } catch (const std::exception &e) {
    GENIEX_LOG_ERROR("failed to create qairt plugin: {}",
                     fmt::to_string(e.what()));
    return nullptr;
  }
}
