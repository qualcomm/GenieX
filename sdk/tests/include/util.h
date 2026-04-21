#pragma once

#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <string_view>  // IWYU pragma: keep
#include <vector>

#include "build_config.h"
#include "doctest.h"
#include "geniex.h"
#include "logging.h"

#define CHECK_ML_ERROR(res) \
    CHECK_MESSAGE((res) >= 0, std::string_view(geniex_get_error_message(static_cast<geniex_ErrorCode>(res))))
#define REQUIRE_ML_ERROR(res) \
    REQUIRE_MESSAGE((res) >= 0, std::string_view(geniex_get_error_message(static_cast<geniex_ErrorCode>(res))))

// Test Summary Infrastructure

// Test summary structure for collecting and reporting test run information
struct TestSummary {
    std::vector<std::string> skipped_models;

    void add_skipped_model(const std::string& model_name, const std::string& model_path) {
        std::string entry = model_name + " (" + model_path + ")";
        skipped_models.push_back(entry);
    }

    void print_summary() const {
        if (skipped_models.empty()) {
            return;
        }

        GENIEX_LOG_WARN("\n========================================");
        GENIEX_LOG_WARN("MODEL SKIP SUMMARY");
        GENIEX_LOG_WARN("========================================");
        GENIEX_LOG_WARN("Total models skipped: {}", skipped_models.size());
        GENIEX_LOG_WARN("Skipped models:");
        for (const auto& model : skipped_models) {
            GENIEX_LOG_WARN("  - {}", model);
        }
        GENIEX_LOG_WARN("========================================\n");
    }

    void reset() { skipped_models.clear(); }
};

// Global test summary instance
inline TestSummary g_test_summary;

// Test Infrastructure

// Test function signature
template <typename M>
using TestFunc = std::function<void(M*, const std::string&)>;

// Collection of test functions for a model type
template <typename M>
class TestRegistry {
   public:
    std::vector<std::pair<std::string, TestFunc<M>>> tests;

    void add(const std::string& name, TestFunc<M> func) { tests.push_back({name, func}); }

    // Run all registered tests for a specific plugin and model index
    template <typename PluginType, typename SetupGuard>
    void run_for_model(SetupGuard& setup_guard, size_t model_idx) {
        std::string model_name = setup_guard.template get_model_name<PluginType>(model_idx);

        // Create a subcase for each test with this model
        for (const auto& [test_name, test_func] : tests) {
            std::string subcase_name = model_name + "/" + test_name;

            SUBCASE(subcase_name.c_str()) {
                // Lazy create model only when this subcase is entered (after doctest filtering)
                // Setup::get() already implements singleton pattern - it caches models and only
                // creates them once per plugin/model_idx combination
                // Note: get() calls reset automatically when returning cached models
                M* model = setup_guard.template get<PluginType>(model_idx);

                // If model is nullptr, it means the model file was not found
                if (model == nullptr) {
                    GENIEX_LOG_WARN("Skipping test '{}' for model '{}' - model file not found", test_name, model_name);
                    return;  // Skip this test
                }

                GENIEX_LOG_INFO("========== Running test: {} for model {} ==========", test_name, model_name);

                // Run the test
                test_func(model, model_name);

                GENIEX_LOG_INFO("========== Completed test: {} ==========", test_name);
            }
        }
    }

    // Run all registered tests for all models of a specific plugin
    template <typename PluginType, typename SetupGuard>
    void run_for_plugin(SetupGuard& setup_guard) {
        size_t model_count = setup_guard.template get_model_count<PluginType>();
        for (size_t model_idx = 0; model_idx < model_count; model_idx++) {
            run_for_model<PluginType>(setup_guard, model_idx);
        }
    }
};

// Macro to define and register a test in one place
// Uses variadic args to handle commas in test body (e.g., initializer lists)
#define REGISTER_TEST(registry, test_name, ...) \
    registry.add(#test_name, [](auto* model, const std::string& model_name) { __VA_ARGS__ })

// Macro to generate TEST_CASE for a single plugin
#define TEST_CASE_FOR_PLUGIN(ModelType, PluginType, setup_guard, register_tests_func) \
    TEST_CASE(#ModelType " " #PluginType* doctest::test_suite(#PluginType)) {         \
        if (setup_guard.get_model_count<PluginType>() > 0) {                          \
            TestRegistry<ModelType> registry;                                         \
            register_tests_func(registry);                                            \
            registry.run_for_plugin<PluginType>(setup_guard);                         \
        }                                                                             \
    }

// Setup

#define PLUGIN_DEF(plugin_name, plugin_id)                                                            \
    inline constexpr auto plugin_name##_str = plugin_id;                                              \
    using plugin_name                       = std::integral_constant<const char*, plugin_name##_str>; \
    TYPE_TO_STRING(plugin_name);

// plugin id map - using string literals for plugin IDs
PLUGIN_DEF(llama_cpp, geniex::build_config::kPluginIdLlamaCpp);
PLUGIN_DEF(qairt, geniex::build_config::kPluginIdQairt);

template <typename P>
using SetupMap = std::map<geniex_PluginId, std::vector<P>>;

template <typename P, typename M>
class Setup {
   private:
    SetupMap<P>                                      param_map;
    std::function<M*(geniex_PluginId, P)>            create_func;
    std::function<int32_t(M*)>                       reset_func;
    std::function<int32_t(M*)>                       destroy_func;
    std::map<std::pair<geniex_PluginId, size_t>, M*> handlers;
    geniex_PluginId                                  current_plugin;
    size_t                                           current_model_idx;
    M*                                               current_handler;  // Track current active handler

   public:
    Setup(SetupMap<P> param_map, std::function<M*(geniex_PluginId, P)> create_func,
        std::function<int32_t(M*)> reset_func, std::function<int32_t(M*)> destroy_func)
        : param_map(param_map),
          create_func(create_func),
          reset_func(reset_func),
          destroy_func(destroy_func),
          current_plugin(nullptr),
          current_model_idx(0),
          current_handler(nullptr) {
        GENIEX_LOG_DEBUG("Setup: Constructor called");
        geniex_init();
    }

    template <typename T>
    size_t get_model_count() const {
        auto it = param_map.find(T::value);
        if (it == param_map.end()) return 0;
        return it->second.size();
    }

    template <typename T>
    std::string get_model_name(size_t idx) const {
        auto it = param_map.find(T::value);
        if (it == param_map.end() || idx >= it->second.size()) {
            return "unknown";
        }
        // Extract name from the first element of the tuple (assuming Param has name as first element)
        return std::get<0>(it->second[idx]);
    }

    // Lazy singleton accessor: creates model on first call, returns cached instance on subsequent calls
    // This enables doctest subcase filtering to skip model creation for filtered-out tests
    template <typename T>
    M* get(size_t model_idx = 0) {
        auto key = std::make_pair(T::value, model_idx);

        // Check if we're switching plugins (not allowed)
        if (current_plugin != nullptr && current_plugin != T::value) {
            GENIEX_LOG_ERROR(
                "use different plugin ({}) with previous ({}), please do not run different plugins in "
                "same time, use ctest or test case filter like -tc=\"*<llama_cpp>\"",
                T::value,
                current_plugin);
            std::abort();
        }

        // Check if we're switching models within the same plugin
        if (current_handler != nullptr && (current_plugin != T::value || current_model_idx != model_idx)) {
            // We're switching to a different model, destroy the current one
            GENIEX_LOG_DEBUG(
                "Switching from model {} to model {}, destroying previous model", current_model_idx, model_idx);

            auto current_key = std::make_pair(current_plugin, current_model_idx);
            auto current_it  = handlers.find(current_key);
            if (current_it != handlers.end()) {
                if (destroy_func && current_it->second) {
                    destroy_func(current_it->second);
                }
                handlers.erase(current_it);
            }
            current_handler = nullptr;
        }

        // Check if this model already exists (cached singleton)
        auto it = handlers.find(key);
        if (it != handlers.end()) {
            // Model exists, just reset and return it
            current_plugin    = T::value;
            current_model_idx = model_idx;
            current_handler   = it->second;
            if (reset_func) reset_func(it->second);
            return it->second;
        }

        // Create new handler (first call for this model)
        const auto& params = param_map.at(T::value);
        if (model_idx >= params.size()) {
            GENIEX_LOG_ERROR("Model index {} out of range for plugin {}", model_idx, T::value);
            std::abort();
        }

        M* handler = create_func(T::value, params[model_idx]);

        // If handler is nullptr, model creation failed (file not found)
        // Don't cache nullptr, just return it to signal skip
        if (handler == nullptr) {
            return nullptr;
        }

        handlers[key]     = handler;
        current_plugin    = T::value;
        current_model_idx = model_idx;
        current_handler   = handler;
        return handler;
    };

    void destroy() {
        GENIEX_LOG_DEBUG("Setup: destroy called");
        if (destroy_func) {
            for (auto& [key, handler] : handlers) {
                if (handler) destroy_func(handler);
            }
        }
        handlers.clear();
        geniex_deinit();
    }

    ~Setup() { GENIEX_LOG_DEBUG("Setup: Destructor called"); }
};

#define TEST_MAIN()                                      \
    int main(int argc, char** argv) {                    \
        int result = doctest::Context(argc, argv).run(); \
        setup_guard.destroy();                           \
        g_test_summary.print_summary();                  \
        g_test_summary.reset();                          \
        return result;                                   \
    }
