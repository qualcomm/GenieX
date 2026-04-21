#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M) M(qairt)
using Param = std::tuple<std::string, std::string, std::string, std::string, std::string>;

Setup<Param, geniex_CV> setup_guard(
    SetupMap<Param>{
        {qairt::value,
            {
#if defined(__ANDROID__)
                {"paddleocr-npu",
                    "paddleocr",
                    "/data/local/tmp/geniex/modelfiles/paddleocr-npu/weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/paddleocr-npu/weights-1-1.nexa",
                    ""},
                {"yolov13-npu",
                    "yolov13",
                    "/data/local/tmp/geniex/modelfiles/yolov13-npu/weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/yolov13-npu/weights-1-1.nexa",
                    ""},
                {"rfdetr-npu",
                    "rfdetr",
                    "/data/local/tmp/geniex/modelfiles/rfdetr-npu/weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/rfdetr-npu/weights-1-1.nexa",
                    ""},
                {"RMBG-2.0-npu",
                    "rmbg-v2",
                    "/data/local/tmp/geniex/modelfiles/RMBG-2.0-npu/weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/RMBG-2.0-npu/weights-1-1.nexa",
                    ""},
                {"table-transformer-detection-npu",
                    "table-transformer",
                    "/data/local/tmp/geniex/modelfiles/table-transformer-detection-npu/"
                    "weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/table-transformer-detection-npu/"
                    "weights-1-1.nexa",
                    ""},
                {"depth-anything-v2-npu",
                    "depth-anything-v2",
                    "/data/local/tmp/geniex/modelfiles/depth-anything-v2-npu/"
                    "weights-1-1.nexa",
                    "/data/local/tmp/geniex/modelfiles/depth-anything-v2-npu/"
                    "weights-1-1.nexa",
                    ""},
#elif defined(_WIN32)
                {"paddleocr-npu",
                    "paddleocr",
                    "modelfiles/qairt/paddleocr-npu/weights-1-1.nexa",
                    "modelfiles/qairt/paddleocr-npu/weights-1-1.nexa",
                    ""},
                {"yolov12-npu",
                    "yolov12",
                    "modelfiles/qairt/yolov12-npu/weights-1-1.nexa",
                    "modelfiles/qairt/yolov12-npu/weights-1-1.nexa",
                    ""},
                {"rfdetr-npu",
                    "rfdetr",
                    "modelfiles/qairt/rfdetr-npu/weights-1-1.nexa",
                    "modelfiles/qairt/rfdetr-npu/weights-1-1.nexa",
                    ""},
                {"RMBG-2.0-npu",
                    "rmbg-v2",
                    "modelfiles/qairt/RMBG-2.0-npu/weights-1-1.nexa",
                    "modelfiles/qairt/RMBG-2.0-npu/weights-1-1.nexa",
                    ""},
                {"Real-ESRGAN-x4plus-npu",
                    "realesrgan",
                    "modelfiles/qairt/Real-ESRGAN-x4plus-npu/weights-1-1.nexa",
                    "modelfiles/qairt/Real-ESRGAN-x4plus-npu/weights-1-1.nexa",
                    ""},
                {"table-transformer-detection-npu",
                    "table-transformer",
                    "modelfiles/qairt/table-transformer-detection-npu/weights-1-1.nexa",
                    "modelfiles/qairt/table-transformer-detection-npu/weights-1-1.nexa",
                    ""},
                {"depth-anything-v2-npu",
                    "depth-anything-v2",
                    "modelfiles/qairt/depth-anything-v2-npu/weights-1-1.nexa",
                    "modelfiles/qairt/depth-anything-v2-npu/weights-1-1.nexa",
                    ""},
#elif defined(__linux__)
                {"convnext-tiny-npu",
                    "convnext",
                    "modelfiles/qairt/convnext-tiny-npu-IoT/weights-1-1.nexa",
                    "modelfiles/qairt/convnext-tiny-npu-IoT/weights-1-1.nexa",
                    ""},
#endif
                // Add more qairt models here as needed
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_CV *cv_model                                            = nullptr;
        auto [test_id, model_name, rec_file, det_file, char_dict_file] = std::move(param);

        if (!std::filesystem::exists(det_file)) {
            GENIEX_LOG_WARN("Detection model file not found: {}", det_file);
            GENIEX_LOG_WARN("Skipping tests for model: {}", model_name);
            g_test_summary.add_skipped_model(model_name, det_file);
            return static_cast<geniex_CV *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_CVCreateInput input{};
        input.model_name            = model_name.c_str();
        input.plugin_id             = plugin;
        input.config.det_model_path = det_file.c_str();
        input.config.rec_model_path = rec_file.c_str();
        input.config.char_dict_path = char_dict_file.c_str();

        int32_t res = geniex_cv_create(&input, &cv_model);
        CHECK_ML_ERROR(res);
        REQUIRE(cv_model != nullptr);

        return cv_model;
    },
    nullptr, geniex_cv_destroy);

std::string test_image_path = "modelfiles/assets/test_image.png";

// Test function definitions
void test_cv_infer_basic(geniex_CV *cv_model, const std::string &model_name) {
    // Check if test image file exists
    if (!std::filesystem::exists(test_image_path)) {
        GENIEX_LOG_ERROR("Test image file not found: {}", test_image_path);
        REQUIRE(false);
    }

    GENIEX_LOG_INFO("Testing CV inference with image: {}", test_image_path);

    // Set up inference input and output
    geniex_CVInferInput input{};
    input.input_image_path = test_image_path.c_str();

    geniex_CVInferOutput output{};
    int32_t              res = geniex_cv_infer(cv_model, &input, &output);
    CHECK_ML_ERROR(res);

    // Verify results structure
    REQUIRE(output.result_count >= 0);

    if (output.result_count > 0) {
        REQUIRE(output.results != nullptr);
        GENIEX_LOG_INFO("Got {} CV results", output.result_count);

        // Process each result
        for (int i = 0; i < output.result_count; i++) {
            const auto &cv_result = output.results[i];

            // Verify confidence score is in valid range
            CHECK(cv_result.confidence >= 0.0f);
            CHECK(cv_result.confidence <= 1.0f);

            // Log result details
            GENIEX_LOG_INFO("Result {}: confidence={}", i, cv_result.confidence);

            // Check text if present (for OCR)
            if (cv_result.text != nullptr) {
                GENIEX_LOG_INFO("  Text: {}", cv_result.text);
                CHECK(utf8::is_valid(cv_result.text, cv_result.text + std::strlen(cv_result.text)));
            }

            // Check class_id if present (for classification)
            if (cv_result.class_id >= 0) {
                GENIEX_LOG_INFO("  Class ID: {}", cv_result.class_id);
            }

            // Check bounding box if present (for detection)
            if (cv_result.bbox.width > 0 && cv_result.bbox.height > 0) {
                GENIEX_LOG_INFO("  Bounding Box: [{}, {}, {}, {}]",
                    cv_result.bbox.x,
                    cv_result.bbox.y,
                    cv_result.bbox.width,
                    cv_result.bbox.height);

                // Verify bounding box coordinates are reasonable
                CHECK(cv_result.bbox.x >= 0.0f);
                CHECK(cv_result.bbox.y >= 0.0f);
                CHECK(cv_result.bbox.width > 0.0f);
                CHECK(cv_result.bbox.height > 0.0f);
            }

            // Check embedding if present (for feature extraction)
            if (cv_result.embedding != nullptr && cv_result.embedding_dim > 0) {
                GENIEX_LOG_INFO("  Embedding dimension: {}", cv_result.embedding_dim);

                // Verify embedding values are reasonable
                for (int j = 0; j < std::min(5, cv_result.embedding_dim); j++) {
                    CHECK(!std::isnan(cv_result.embedding[j]));
                    CHECK(!std::isinf(cv_result.embedding[j]));
                }
            }
        }
    } else {
        GENIEX_LOG_INFO("No CV results found - this may be normal for some models");
    }

    // Clean up results
    if (output.results) {
        for (int i = 0; i < output.result_count; i++) {
            if (output.results[i].text) {
                geniex_free(const_cast<char *>(output.results[i].text));
            }
            if (output.results[i].embedding) {
                geniex_free(output.results[i].embedding);
            }
        }
        geniex_free(output.results);
    }
}

// Register all CV tests
template <typename PluginType>
void register_cv_tests(TestRegistry<geniex_CV> &registry) {
    REGISTER_TEST(registry, CvInferBasic, test_cv_infer_basic(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_CV, Plugin, setup_guard, register_cv_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()
