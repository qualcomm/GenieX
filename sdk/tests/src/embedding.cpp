#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>
#include <optional>
#include <string>

#include "doctest.h"
#include "geniex.h"
#include "logging.h"
#include "utf8.h"  // IWYU pragma: export
#include "util.h"

namespace {

#define PLUGINS(M) M(llama_cpp) M(qairt)
using Param = std::tuple<std::string, std::string, std::string, std::optional<std::string>, std::optional<std::string>>;

Setup<Param, geniex_Embedder> setup_guard(
    SetupMap<Param>{
        {llama_cpp::value,
            std::vector<Param>{
                {"jina-v2-small-Q4",
                    "jina-v2-small-Q4",
                    "modelfiles/llama_cpp/jina-embeddings-v2-small-en-Q4_K_M.gguf",
                    std::nullopt,
                    std::nullopt},
                // Add more llama_cpp models here as needed
            }},
        {qairt::value,
            std::vector<Param>{
#if defined(__ANDROID__)
                {"embeddinggemma-300m-npu",
                    "embed-gemma",
                    "/data/local/tmp/geniex/modelfiles/embeddinggemma-300m-npu/"
                    "weights-1-2.nexa",
                    std::nullopt,
                    std::nullopt},
                {"embedneural-npu",
                    "embedneural",
                    "/data/local/tmp/geniex/modelfiles/embedneural-npu/"
                    "weights-1-4.nexa",
                    std::nullopt,
                    std::nullopt},
#elif defined(_WIN32)
                {"embeddinggemma-300m-npu",
                    "embed-gemma",
                    "modelfiles/qairt/embeddinggemma-300m-npu/weights-1-2.nexa",
                    std::nullopt,
                    std::nullopt},
                {"EmbedNeural",
                    "embedneural",
                    "modelfiles/qairt/EmbedNeural/weights-1-4.nexa",
                    std::nullopt,
                    std::nullopt},
#else
// QAIRT tests are not supported on this platform (Linux/other)
#endif
            }},
    },
    [](geniex_PluginId plugin, Param param) {
        geniex_Embedder *embedder                                      = nullptr;
        auto [test_id, model_name, model_path, mmproj_path, tokenizer] = std::move(param);

        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            GENIEX_LOG_WARN("Model file not found: {}", model_path);
            GENIEX_LOG_WARN("Skipping tests for model: {}", model_name);
            g_test_summary.add_skipped_model(model_name, model_path);
            return static_cast<geniex_Embedder *>(nullptr);  // Return nullptr to indicate skip
        }

        if (mmproj_path.has_value() && !std::filesystem::exists(mmproj_path.value())) {
            GENIEX_LOG_WARN("MMProj file not found: {}", mmproj_path.value());
            GENIEX_LOG_WARN("Skipping tests for model: {}", model_name);
            g_test_summary.add_skipped_model(model_name, mmproj_path.value());
            return static_cast<geniex_Embedder *>(nullptr);  // Return nullptr to indicate skip
        }

        geniex_EmbedderCreateInput input{};

        // Initialize all fields to zero/nullptr first
        std::memset(&input, 0, sizeof(input));

        // Set model_name if provided (required for qairt plugin)
        input.model_name     = model_name.c_str();
        input.model_path     = model_path.c_str();
        input.mmproj_path    = mmproj_path.has_value() ? mmproj_path.value().c_str() : nullptr;
        input.tokenizer_path = tokenizer.has_value() ? tokenizer.value().c_str() : nullptr;
        input.plugin_id      = plugin;

        // Initialize model config properly for QAIRT
        if (std::string(plugin) == "qairt") {
            input.config.max_tokens = 1024;
            input.config.verbose    = false;
            // QAIRT paths will be injected by the plugin itself
        }

        int32_t res = geniex_embedder_create(&input, &embedder);

        CHECK_ML_ERROR(res);
        REQUIRE(embedder != nullptr);
        return embedder;
    },
    nullptr, geniex_embedder_destroy);

std::string test_text = "🥳 🎂 Once upon a time";

// Test function definitions
void test_embedder_creation(geniex_Embedder *embedder, const std::string &model_name) {
    geniex_EmbedderDimOutput dim_output{};
    int32_t                  res = geniex_embedder_embedding_dim(embedder, &dim_output);
    CHECK_ML_ERROR(res);
    CHECK(dim_output.dimension > 0);
    GENIEX_LOG_INFO("Embedding dimension: {}", dim_output.dimension);
}

void test_embedder_single_text(geniex_Embedder *embedder, const std::string &model_name) {
    // Initialize config properly
    geniex_EmbeddingConfig cfg{};
    cfg.batch_size       = 32;
    cfg.normalize        = true;
    cfg.normalize_method = "l2";

    const char               *texts[] = {test_text.c_str()};
    geniex_EmbedderEmbedInput input{};
    std::memset(&input, 0, sizeof(input));  // Initialize all fields to zero
    input.texts                 = texts;
    input.text_count            = 1;
    input.config                = &cfg;
    input.task_type             = nullptr;
    input.input_ids_2d          = nullptr;
    input.input_ids_row_lengths = nullptr;
    input.input_ids_row_count   = 0;

    geniex_EmbedderEmbedOutput output{};
    std::memset(&output, 0, sizeof(output));
    int32_t res = geniex_embedder_embed(embedder, &input, &output);

    CHECK_ML_ERROR(res);
    REQUIRE(output.embeddings != nullptr);
    CHECK(output.embedding_count == 1);

    // Get embedding dimension for validation
    geniex_EmbedderDimOutput dim_output{};
    geniex_embedder_embedding_dim(embedder, &dim_output);
    int32_t expected_total_floats = dim_output.dimension * output.embedding_count;

    GENIEX_LOG_INFO("{}", output);
    GENIEX_LOG_INFO("Generated {} embeddings with dimension {}", output.embedding_count, dim_output.dimension);

    // Print first 20 embedding values for debugging
    if (output.embeddings != nullptr && expected_total_floats > 0) {
        // Construct string with first 20 embedding values
        std::string embedding_values;
        int         count = std::min(20, expected_total_floats);
        for (int i = 0; i < count; i++) {
            embedding_values += std::to_string(output.embeddings[i]);
            if (i < count - 1) embedding_values += ", ";
        }
        if (expected_total_floats > 20) embedding_values += ", ...";

        GENIEX_LOG_INFO("First 20 embedding values: {}", embedding_values);

        // Calculate and print stats
        float mean =
            std::accumulate(output.embeddings, output.embeddings + expected_total_floats, 0.0f) / expected_total_floats;
        float variance = 0.0f;
        for (int i = 0; i < expected_total_floats; i++) {
            float diff = output.embeddings[i] - mean;
            variance += diff * diff;
        }
        variance /= expected_total_floats;
        float std_dev = std::sqrt(variance);

        GENIEX_LOG_INFO("Embedding stats: min={}, max={}, mean={}, std={}",
            *std::min_element(output.embeddings, output.embeddings + expected_total_floats),
            *std::max_element(output.embeddings, output.embeddings + expected_total_floats),
            mean,
            std_dev);
    }

    if (output.embeddings) {
        geniex_free(output.embeddings);
    }
}

void test_embedder_semantic_similarity(geniex_Embedder *embedder, const std::string &model_name) {
    geniex_EmbeddingConfig cfg{};
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.batch_size       = 1;
    cfg.normalize        = true;
    cfg.normalize_method = "l2";

    std::string text_a_str = "the cat is resting on the rug";
    std::string text_b_str = "a feline sits on the mat";
    std::string text_c_str = "the international space station orbits Earth";

    const char *text_a[] = {text_a_str.c_str()};
    const char *text_b[] = {text_b_str.c_str()};
    const char *text_c[] = {text_c_str.c_str()};

    // Generate embeddings separately
    geniex_EmbedderEmbedInput input_a{};
    input_a.texts      = text_a;
    input_a.text_count = 1;
    input_a.config     = &cfg;
    input_a.task_type  = nullptr;

    geniex_EmbedderEmbedInput input_b{};
    std::memset(&input_b, 0, sizeof(input_b));
    input_b.texts      = text_b;
    input_b.text_count = 1;
    input_b.config     = &cfg;
    input_b.task_type  = nullptr;

    geniex_EmbedderEmbedInput input_c{};
    std::memset(&input_c, 0, sizeof(input_c));
    input_c.texts      = text_c;
    input_c.text_count = 1;
    input_c.config     = &cfg;
    input_c.task_type  = nullptr;

    geniex_EmbedderEmbedOutput output_a{};
    geniex_EmbedderEmbedOutput output_b{};
    geniex_EmbedderEmbedOutput output_c{};

    int32_t res_a = geniex_embedder_embed(embedder, &input_a, &output_a);
    int32_t res_b = geniex_embedder_embed(embedder, &input_b, &output_b);
    int32_t res_c = geniex_embedder_embed(embedder, &input_c, &output_c);

    CHECK_ML_ERROR(res_a);
    CHECK_ML_ERROR(res_b);
    CHECK_ML_ERROR(res_c);
    REQUIRE(output_a.embeddings != nullptr);
    REQUIRE(output_b.embeddings != nullptr);
    REQUIRE(output_c.embeddings != nullptr);

    // Get embedding dimension
    geniex_EmbedderDimOutput dim_output{};
    geniex_embedder_embedding_dim(embedder, &dim_output);
    int32_t dim = dim_output.dimension;

    // Helper function to calculate cosine similarity between two vectors
    auto cosine_similarity = [](const float *vec_a, const float *vec_b, int dim) {
        float dot_product = 0.0f;
        float norm_a      = 0.0f;
        float norm_b      = 0.0f;
        for (int i = 0; i < dim; ++i) {
            dot_product += vec_a[i] * vec_b[i];
            norm_a += vec_a[i] * vec_a[i];
            norm_b += vec_b[i] * vec_b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f) {
            return 0.0f;  // Avoid division by zero
        }

        return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
    };

    float sim_ab = cosine_similarity(output_a.embeddings, output_b.embeddings,
        dim);  // Similar pair
    float sim_ac = cosine_similarity(output_a.embeddings, output_c.embeddings,
        dim);  // Different pair

    GENIEX_LOG_INFO("Similarity(cat, feline): {}", sim_ab);
    GENIEX_LOG_INFO("Similarity(cat, space station): {}", sim_ac);

    // Similar texts should have higher similarity than different texts
    CHECK(sim_ab > sim_ac);

    // Clean up
    if (output_a.embeddings) geniex_free(output_a.embeddings);
    if (output_b.embeddings) geniex_free(output_b.embeddings);
    if (output_c.embeddings) geniex_free(output_c.embeddings);
}

void test_embedder_batch_processing(geniex_Embedder *embedder, const std::string &model_name) {
    geniex_EmbeddingConfig cfg{};
    cfg.batch_size       = 4;
    cfg.normalize        = true;
    cfg.normalize_method = "l2";

    std::vector<std::string> text_strs = {
        "Hello world", "Good morning", "Machine learning is fascinating", "Natural language processing"};
    std::vector<const char *> text_ptrs;
    for (const auto &text : text_strs) {
        text_ptrs.push_back(text.c_str());
    }

    geniex_EmbedderEmbedInput input{};
    std::memset(&input, 0, sizeof(input));
    input.texts      = text_ptrs.data();
    input.text_count = 4;
    input.config     = &cfg;
    input.task_type  = nullptr;

    geniex_EmbedderEmbedOutput output{};
    std::memset(&output, 0, sizeof(output));
    int32_t res = geniex_embedder_embed(embedder, &input, &output);

    CHECK_ML_ERROR(res);
    REQUIRE(output.embeddings != nullptr);
    CHECK(output.embedding_count == 4);

    GENIEX_LOG_INFO("Batch processing: {}", output);

    if (output.embeddings) {
        geniex_free(output.embeddings);
    }
}

void test_image_search(geniex_Embedder *embedder, const std::string &model_name) {
    GENIEX_LOG_INFO("Running image search test for multimodal model: {}", model_name);

    // Config without normalization
    geniex_EmbeddingConfig cfg{};
    cfg.batch_size       = 1;
    cfg.normalize        = false;
    cfg.normalize_method = "none";

    // Embed text query: "A blue cat"
    const char               *query_text[] = {"A blue cat"};
    geniex_EmbedderEmbedInput text_input{};
    std::memset(&text_input, 0, sizeof(text_input));
    text_input.texts      = query_text;
    text_input.text_count = 1;
    text_input.config     = &cfg;

    geniex_EmbedderEmbedOutput text_output{};
    std::memset(&text_output, 0, sizeof(text_output));
    int32_t res = geniex_embedder_embed(embedder, &text_input, &text_output);
    CHECK_ML_ERROR(res);
    REQUIRE(text_output.embeddings != nullptr);
    CHECK(text_output.embedding_count == 1);

    // Embed blue cat image
    const char               *blue_cat_path[] = {"modelfiles/assets/blue_cat.jpg"};
    geniex_EmbedderEmbedInput blue_cat_input{};
    std::memset(&blue_cat_input, 0, sizeof(blue_cat_input));
    blue_cat_input.image_paths = const_cast<geniex_Path *>(blue_cat_path);
    blue_cat_input.image_count = 1;
    blue_cat_input.config      = &cfg;

    geniex_EmbedderEmbedOutput blue_cat_output{};
    std::memset(&blue_cat_output, 0, sizeof(blue_cat_output));
    res = geniex_embedder_embed(embedder, &blue_cat_input, &blue_cat_output);
    CHECK_ML_ERROR(res);
    REQUIRE(blue_cat_output.embeddings != nullptr);
    CHECK(blue_cat_output.embedding_count == 1);

    // Embed red cat image
    const char               *red_cat_path[] = {"modelfiles/assets/red_cat.jpg"};
    geniex_EmbedderEmbedInput red_cat_input{};
    std::memset(&red_cat_input, 0, sizeof(red_cat_input));
    red_cat_input.image_paths = const_cast<geniex_Path *>(red_cat_path);
    red_cat_input.image_count = 1;
    red_cat_input.config      = &cfg;

    geniex_EmbedderEmbedOutput red_cat_output{};
    std::memset(&red_cat_output, 0, sizeof(red_cat_output));
    res = geniex_embedder_embed(embedder, &red_cat_input, &red_cat_output);
    CHECK_ML_ERROR(res);
    REQUIRE(red_cat_output.embeddings != nullptr);
    CHECK(red_cat_output.embedding_count == 1);

    // Get embedding dimension
    geniex_EmbedderDimOutput dim_output{};
    geniex_embedder_embedding_dim(embedder, &dim_output);
    int32_t dim = dim_output.dimension;

    // Calculate Manhattan distance (L1 distance)
    auto manhattan_distance = [](const float *vec_a, const float *vec_b, int dim) {
        float distance = 0.0f;
        for (int i = 0; i < dim; ++i) {
            distance += std::abs(vec_a[i] - vec_b[i]);
        }
        return distance;
    };

    float dist_query_blue = manhattan_distance(text_output.embeddings, blue_cat_output.embeddings, dim);
    float dist_query_red  = manhattan_distance(text_output.embeddings, red_cat_output.embeddings, dim);

    GENIEX_LOG_INFO("Manhattan distance (query 'A blue cat' -> blue_cat.jpg): {}", dist_query_blue);
    GENIEX_LOG_INFO("Manhattan distance (query 'A blue cat' -> red_cat.jpg): {}", dist_query_red);

    // Assert that the query has lower distance with blue cat than red cat
    CHECK(dist_query_blue < dist_query_red);

    // Clean up
    if (text_output.embeddings) geniex_free(text_output.embeddings);
    if (blue_cat_output.embeddings) geniex_free(blue_cat_output.embeddings);
    if (red_cat_output.embeddings) geniex_free(red_cat_output.embeddings);
}

void test_embedder_video(geniex_Embedder *embedder, const std::string &model_name) {
    GENIEX_LOG_INFO("Running video embedding test for multimodal model: {}", model_name);

    const std::string video_path = "modelfiles/assets/test_video.mp4";
    if (!std::filesystem::exists(video_path)) {
        GENIEX_LOG_WARN("Video file not found: {}", video_path);
        GENIEX_LOG_WARN("Skipping video embedding test for model: {}", model_name);
        g_test_summary.add_skipped_model(model_name, video_path);
        return;
    }

    geniex_EmbeddingConfig cfg{};
    cfg.batch_size       = 1;
    cfg.normalize        = true;
    cfg.normalize_method = "l2";

    const char *video_paths[]  = {video_path.c_str()};
    float       video_starts[] = {0.0f};
    float       video_ends[]   = {8.0f};

    geniex_EmbedderEmbedInput input{};
    std::memset(&input, 0, sizeof(input));
    input.video_paths  = const_cast<geniex_Path *>(video_paths);
    input.video_starts = video_starts;
    input.video_ends   = video_ends;
    input.video_count  = 1;
    input.config       = &cfg;

    geniex_EmbedderEmbedOutput output{};
    std::memset(&output, 0, sizeof(output));
    int32_t res = geniex_embedder_embed(embedder, &input, &output);

    CHECK_ML_ERROR(res);
    REQUIRE(output.embeddings != nullptr);
    CHECK(output.embedding_count == 1);

    if (output.embeddings) {
        geniex_free(output.embeddings);
    }
}

// Register all embedder tests
template <typename PluginType>
void register_embedder_tests(TestRegistry<geniex_Embedder> &registry) {
    REGISTER_TEST(registry, EmbedderCreation, test_embedder_creation(model, model_name););
    REGISTER_TEST(registry, EmbedderSingleText, test_embedder_single_text(model, model_name););
    REGISTER_TEST(registry, EmbedderSemanticSimilarity, test_embedder_semantic_similarity(model, model_name););
    REGISTER_TEST(registry, EmbedderBatchProcessing, test_embedder_batch_processing(model, model_name););
}

// Generate test cases for all plugins
#define GEN(Plugin) TEST_CASE_FOR_PLUGIN(geniex_Embedder, Plugin, setup_guard, register_embedder_tests<Plugin>)
PLUGINS(GEN)
#undef GEN

}  // namespace

TEST_MAIN()