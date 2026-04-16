#include "doctest.h"
#include "logging.h"
#include "ml.h"
#include "util.h"

Setup<int, int> setup_guard(SetupMap<int>{}, nullptr, nullptr, nullptr);

TEST_CASE("GetPluginList") {
    ml_GetPluginListOutput output{};
    CHECK_ML_ERROR(ml_get_plugin_list(&output));
    for (int32_t i = 0; i < output.plugin_count; i++) {
        GENIEX_LOG_INFO("found plugin id: {}", output.plugin_ids[i]);
    }
    ml_free(output.plugin_ids);
}

TEST_CASE("GetDeviceList") {
    ml_GetPluginListOutput plugin_output{};
    CHECK_ML_ERROR(ml_get_plugin_list(&plugin_output));
    REQUIRE(plugin_output.plugin_count > 0);
    REQUIRE(plugin_output.plugin_ids != nullptr);

    ml_GetDeviceListInput input{};
    input.plugin_id = plugin_output.plugin_ids[0];

    ml_GetDeviceListOutput output{};
    CHECK_ML_ERROR(ml_get_device_list(&input, &output));
    for (int32_t i = 0; i < output.device_count; i++) {
        GENIEX_LOG_INFO("found device: '{}'->'{}'", output.device_ids[i], output.device_names[i]);
    }
    ml_free(output.device_ids);
    ml_free(output.device_names);
    ml_free(plugin_output.plugin_ids);
}

// Cannot test on ci, as it requires a specific plugin to be registered
// #include <dlfcn.h>
//
// TEST_CASE("RegisterPlugin") {
//     ml_GetPluginListOutput plugin_output{};
//     CHECK_ML_ERROR(ml_get_plugin_list(&plugin_output));
//     GENIEX_LOG_INFO("Registered plugins: {}", plugin_output);
//
//     auto handle = dlopen("/home/remilia/Workspace/github/geniex-bridge/build/out/geniex_llama_cpp/libgeniex_plugin.so",
//         RTLD_NOW | RTLD_LOCAL);
//     if (!handle) {
//         throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
//     }
//
//     auto plugin_id   = dlsym(handle, "plugin_id");
//     auto create_func = dlsym(handle, "create_plugin");
//
//     CHECK_ML_ERROR(ml_register_plugin(plugin_id, create_func));
//     CHECK_ML_ERROR(ml_get_plugin_list(&plugin_output));
//     CHECK(plugin_output.plugin_count > 0);
//     CHECK(plugin_output.plugin_ids != nullptr);
//
//     ml_LLM*           llm = nullptr;
//     ml_LlmCreateInput input{};
//     input.model_path       = "modelfiles/llama_cpp/Qwen3-0.6B-Q8_0.gguf";
//     input.tokenizer_path   = nullptr;
//     input.config.n_ctx     = 512;
//     input.config.n_seq_max = 64;
//     input.plugin_id        = "llama_cpp";
//     CHECK_ML_ERROR(ml_llm_create(&input, &llm));
//     REQUIRE(llm != nullptr);
// }

TEST_MAIN()
