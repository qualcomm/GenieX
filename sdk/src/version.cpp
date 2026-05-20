#include "version.h"

#include "build_config.h"

#ifdef GENIEX_PLUGIN_LLAMA_CPP
extern const char* LLAMA_COMMIT;
#endif

namespace geniex::build_config {
const char kBridgeVersion[] = GENIEX_VERSION;
const char kQairtVersion[]  = GENIEX_QAIRT_VERSION;
#ifdef GENIEX_PLUGIN_LLAMA_CPP
const char* const kLlamaCppVersion = LLAMA_COMMIT;
#else
const char* const kLlamaCppVersion = "unavailable";
#endif
}  // namespace geniex::build_config
