#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>, std::optional<std::string>>;

Setup<Param, geniex_ImageGen> setup_guard(
    SetupMap<Param>{},
    [](geniex_PluginId plugin, Param param) {
        geniex_ImageGen *imagegen                      = nullptr;
        auto [test_id, name, model, scheduler, device] = std::move(param);

        // Check if model path exists (it's a directory for image generation
        // models)
        if (!std::filesystem::exists(model)) {
            GENIEX_LOG_WARN("Model path not found: {}", model);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model);
            return static_cast<geniex_ImageGen *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_ImageGenCreateInput input{};
        input.model_name            = name.c_str();
        input.model_path            = model.c_str();
        input.scheduler_config_path = scheduler.has_value() ? scheduler.value().c_str() : nullptr;
        input.plugin_id             = plugin;
        input.device_id             = device.has_value() ? device.value().c_str() : nullptr;
        int32_t res                 = geniex_imagegen_create(&input, &imagegen);
        REQUIRE_ML_ERROR(res);

        REQUIRE(imagegen != nullptr);
        return imagegen;
    },
    nullptr, geniex_imagegen_destroy);

std::string test_prompt     = "a beautiful landscape with mountains and trees";
std::string negative_prompt = "blurry, low quality, distorted";

// Test function definitions
void test_imagegen_txt2img(geniex_ImageGen *imagegen, const std::string &model_name) {
    // The image generator must be created successfully or the test fails
    REQUIRE(imagegen != nullptr);

    // Set up image generation configuration (aligned with config.json)
    geniex_ImageSamplerConfig sampler_config = {};
    sampler_config.method                    = "ddim";
    sampler_config.steps                     = 20;
    sampler_config.guidance_scale            = 7.5f;
    sampler_config.eta                       = 0.0f;
    sampler_config.seed                      = 2;  // Changed from 42 to match config.json

    geniex_SchedulerConfig scheduler_config = {};
    scheduler_config.type                   = "ddim";
    scheduler_config.num_train_timesteps    = 1000;
    scheduler_config.steps_offset           = 1;
    scheduler_config.beta_start             = 0.00085f;
    scheduler_config.beta_end               = 0.012f;
    scheduler_config.beta_schedule          = "scaled_linear";
    scheduler_config.prediction_type        = "epsilon";
    scheduler_config.timestep_type          = "discrete";
    scheduler_config.timestep_spacing       = "leading";
    scheduler_config.interpolation_type     = "linear";
    scheduler_config.config_path            = nullptr;

    const char *prompts[]     = {test_prompt.c_str()};
    const char *neg_prompts[] = {negative_prompt.c_str()};

    geniex_ImageGenerationConfig config = {};
    config.prompts                      = prompts;
    config.prompt_count                 = 1;
    config.negative_prompts             = neg_prompts;
    config.negative_prompt_count        = 1;
    config.height                       = 512;
    config.width                        = 512;
    config.sampler_config               = sampler_config;
    config.scheduler_config             = scheduler_config;
    config.strength                     = 1.0f;

    geniex_ImageGenTxt2ImgInput input = {};
    input.prompt_utf8                 = test_prompt.c_str();
    input.config                      = &config;
    input.output_path                 = "./build/output/generated_image.png";

    if (std::filesystem::exists(input.output_path)) {
        std::filesystem::remove(input.output_path);
    }

    geniex_ImageGenOutput output = {};
    int32_t               res    = geniex_imagegen_txt2img(imagegen, &input, &output);
    CHECK_ML_ERROR(res);

    GENIEX_LOG_INFO("{}", output);
    CHECK(output.output_image_path != nullptr);
    CHECK(std::filesystem::exists(output.output_image_path));
    CHECK(std::filesystem::file_size(output.output_image_path) > 0);
}

// Register all image generation tests
template <typename PluginType>
void register_imagegen_tests(TestRegistry<geniex_ImageGen> &registry) {
    REGISTER_TEST(registry, ImageGenTxt2Img, test_imagegen_txt2img(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_ImageGen, Plugin, setup_guard, register_imagegen_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
