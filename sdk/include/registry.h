#pragma once

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "build_config.h"
#include "plugin/ICv.h"
#include "plugin/IEmbedding.h"
#include "plugin/IImageGen.h"
#include "plugin/ITts.h"
#include "plugin/Plugin.h"

namespace geniex {

using PluginId = std::string;

class PluginFactory {
   private:
    void*                    handle = nullptr;
    std::function<Plugin*()> create_func;
    std::unique_ptr<Plugin>  cached_plugin;  // internal cache, manage lifecycle
    std::string              plugin_id;      // plugin identifier

    void load_library(const std::filesystem::path& path);
    void load_symbol(const std::string& symbol_name, void*& symbol_ptr);
    void close_library();

   public:
    PluginFactory(const std::filesystem::path& path = "");
    PluginFactory(void* id, void* create_func);
    ~PluginFactory();
    PluginFactory(const PluginFactory&)            = delete;
    PluginFactory& operator=(const PluginFactory&) = delete;

    Plugin*         get_instance();  // return raw pointer, but internal manage lifecycle
    const PluginId& get_plugin_id() const { return plugin_id; }
};

class PluginLoadException : public std::exception {};
class PluginNotFoundException : public std::exception {};

class Registry {
   private:
    // PluginId -> PluginFactory: use std::string compare to avoid type mismatch
    mutable std::unordered_map<PluginId, std::unique_ptr<PluginFactory>> plugins;
    mutable std::vector<std::string>                                     failed_plugins;
    mutable std::mutex                                                   mutex;

    Registry()  = default;
    ~Registry() = default;

   public:
    static Registry& instance();
    void             scan_plugins();
    void             register_plugin(void* plugin_id_func, void* create_func);
    void             clear();

    template <typename M>
    M* get(const char* type) {
        return get<M>(std::string(type));
    }

    template <typename M>
    M* get(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex);

        std::string plugin_id = id;

        // compact previous plugin id
        std::map<std::string, std::string> id_map = {
            {"llama_cpp", build_config::kPluginIdLlamaCpp},
        };
        auto map_it = id_map.find(plugin_id);
        if (map_it != id_map.end()) {
            GENIEX_LOG_WARN("Plugin id '{}' is deprecated, use '{}' instead", id, map_it->second);
            plugin_id = map_it->second;
        }

        auto it = plugins.find(plugin_id);
        if (it == plugins.end()) {
            // check if it's in failed plugins
            for (const auto& failed_plugin : failed_plugins) {
                GENIEX_LOG_DEBUG("Checking failed plugin: {}", failed_plugin);
                if (failed_plugin == plugin_id) {
                    throw PluginLoadException();
                }
            }
            throw PluginNotFoundException();
        }

        Plugin* plugin = it->second->get_instance();
        if constexpr (std::is_same_v<M, Plugin>) {
            return plugin;
        } else if constexpr (std::is_same_v<M, ILlm>) {
            return plugin->create_llm();
        } else if constexpr (std::is_same_v<M, IVlm>) {
            return plugin->create_vlm();
        } else if constexpr (std::is_same_v<M, IEmbedding>) {
            return plugin->create_embedding();
        } else if constexpr (std::is_same_v<M, IAsr>) {
            return plugin->create_asr();
        } else if constexpr (std::is_same_v<M, ITts>) {
            return plugin->create_tts();
        } else if constexpr (std::is_same_v<M, ICv>) {
            return plugin->create_cv();
        } else if constexpr (std::is_same_v<M, IReranker>) {
            return plugin->create_reranker();
        } else if constexpr (std::is_same_v<M, IImageGen>) {
            return plugin->create_image_gen();
        } else if constexpr (std::is_same_v<M, IDiarize>) {
            return plugin->create_diarize();
        } else {
            throw std::runtime_error("Unsupported modality type");
        }
    }

    std::vector<std::string> get_plugin_list() {
        std::lock_guard<std::mutex> lock(mutex);
        std::vector<std::string>    plugin_list;
        for (auto& [plugin_id, factory] : plugins) {
            plugin_list.push_back(plugin_id);
        }
        return plugin_list;
    }
};

}  // namespace geniex
