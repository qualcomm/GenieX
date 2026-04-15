#include "plugin/Plugin.h"

#include <cstdlib>
#include <exception>

#include "build_config.h"
#include "llm.h"
#include "vlm.h"
#include "logging.h"

namespace geniex {

class QairtPlugin : public Plugin {
   public:
    QairtPlugin() {
        GENIEX_LOG_TRACE("creating and initializing qairt plugin");
    }

    ~QairtPlugin() override {
        GENIEX_LOG_TRACE("destroying qairt plugin");
    }

    int32_t get_device_list(const ml_GetDeviceListInput* input,
                            ml_GetDeviceListOutput* output) override {
        if (!input || !output) {
            return ML_ERROR_COMMON_INVALID_INPUT;
        }

        static const char* device_ids[] = {"NPU"};
        static const char* device_names[] = {"Qualcomm NPU (QAIRT)"};

        output->device_ids = device_ids;
        output->device_names = device_names;
        output->device_count = 1;
        return ML_SUCCESS;
    }

    ILlm* create_llm() override { return new geniex::QairtLlm; }
    IVlm* create_vlm() override { return new geniex::QairtVlm; }
};

}  // namespace geniex

#ifdef GENIEX_STATIC

#else

ml_PluginId plugin_id() { return geniex::build_config::kPluginIdQairt; }

geniex::Plugin* create_plugin() {
    try {
        GENIEX_LOG_TRACE("creating qairt plugin");
        return new geniex::QairtPlugin;
    } catch (const std::exception& e) {
        GENIEX_LOG_ERROR("failed to create qairt plugin: {}", e.what());
        return nullptr;
    }
}

#endif
