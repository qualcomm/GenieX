#include <filesystem>
#include <optional>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "util.h"

namespace {

// rerank supported by llama_cpp and QAIRT in this repository profile
#define PLUGINS(M) M(llama_cpp) M(qairt)
using Param = std::tuple<std::string, std::string, std::optional<std::string>>;

Setup<Param, geniex_Reranker> setup_guard(
    SetupMap<Param>{
        {llama_cpp::value,
            {
                {"bge-reranker-v2-m3", "modelfiles/llama_cpp/bge-reranker-v2-m3-Q4_K_M.gguf", std::nullopt},
                // Add more llama_cpp models here as needed
            }},
        {qairt::value,
            {
#if defined(__ANDROID__)
                {"jina-rerank",
                    "/data/local/tmp/geniex/modelfiles/jina-rerank-npu/"
                    "weights-1-4.nexa",
                    std::nullopt},
#elif defined(_WIN32)
                {"jina-rerank", "modelfiles/qairt/jina-rerank-npu/weights-1-4.nexa", std::nullopt},
#endif
                // Add more qairt models here as needed
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_Reranker *reranker               = nullptr;
        auto [name, model_path, tokenizer_path] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            GENIEX_LOG_WARN("Model file not found: {}", model_path);
            GENIEX_LOG_WARN("Skipping tests for model: {}", name);
            g_test_summary.add_skipped_model(name, model_path);
            return static_cast<geniex_Reranker *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_RerankerCreateInput input{};
        input.model_name     = name.c_str();
        input.model_path     = model_path.c_str();
        input.tokenizer_path = tokenizer_path ? tokenizer_path->c_str() : nullptr;
        input.plugin_id      = plugin;
        int32_t res          = geniex_reranker_create(&input, &reranker);
        CHECK_ML_ERROR(res);
        REQUIRE(reranker != nullptr);

        return reranker;
    },
    nullptr, geniex_reranker_destroy);

std::string              test_query     = "What is machine learning?";
std::vector<std::string> test_documents = {
    "Machine learning is a subset of artificial intelligence that enables "
    "computers to learn and make decisions "
    "without being explicitly programmed.",
    "Machine learning algorithms build mathematical models based on sample "
    "data to make predictions or decisions.",
    "Deep learning is a subset of machine learning that uses neural networks "
    "with multiple layers.",
    "Python is a popular programming language for machine learning and data "
    "science.",
    "The weather today is sunny and warm."};

// Test function definitions
void test_reranker_single_query(geniex_Reranker *reranker, const std::string &model_name) {
    geniex_RerankConfig cfg = {};
    cfg.batch_size          = 32;
    cfg.normalize           = true;
    cfg.normalize_method    = "l2";

    std::vector<const char *> documents(test_documents.size());
    for (size_t i = 0; i < test_documents.size(); ++i) {
        documents[i] = test_documents[i].c_str();
    }

    geniex_RerankerRerankInput input = {};
    input.query                      = test_query.c_str();
    input.documents                  = documents.data();
    input.documents_count            = static_cast<int32_t>(test_documents.size());
    input.config                     = &cfg;

    geniex_RerankerRerankOutput output = {};
    int32_t                     res    = geniex_reranker_rerank(reranker, &input, &output);

    CHECK_ML_ERROR(res);
    REQUIRE(output.scores != nullptr);
    CHECK(output.score_count == test_documents.size());

    GENIEX_LOG_INFO("{}", output);
    for (int32_t i = 0; i < output.score_count; ++i) {
        GENIEX_LOG_INFO("Score {}: {}", i, output.scores[i]);
    }
    GENIEX_LOG_INFO("Profile data: {}", output.profile_data);
}

void test_reranker_batch_processing(geniex_Reranker *reranker, const std::string &model_name) {
    geniex_RerankConfig cfg = {};
    cfg.batch_size          = 2;  // Small batch size for testing
    cfg.normalize           = true;

    std::vector<const char *> documents(test_documents.size());
    for (size_t i = 0; i < test_documents.size(); ++i) {
        documents[i] = test_documents[i].c_str();
    }

    geniex_RerankerRerankInput input = {};
    input.query                      = test_query.c_str();
    input.documents                  = documents.data();
    input.documents_count            = static_cast<int32_t>(test_documents.size());
    input.config                     = &cfg;

    geniex_RerankerRerankOutput output = {};
    int32_t                     res    = geniex_reranker_rerank(reranker, &input, &output);

    CHECK_ML_ERROR(res);
    REQUIRE(output.scores != nullptr);
    CHECK(output.score_count == test_documents.size());

    GENIEX_LOG_INFO("{}", output);
    for (int32_t i = 0; i < output.score_count; ++i) {
        GENIEX_LOG_INFO("Score {}: {}", i, output.scores[i]);
    }
    GENIEX_LOG_INFO("Profile data: {}", output.profile_data);
}

// Register all reranker tests
template <typename PluginType>
void register_reranker_tests(TestRegistry<geniex_Reranker> &registry) {
    REGISTER_TEST(registry, RerankerSingleQuery, test_reranker_single_query(model, model_name););
    REGISTER_TEST(registry, RerankerBatchProcessing, test_reranker_batch_processing(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_Reranker, Plugin, setup_guard, register_reranker_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()