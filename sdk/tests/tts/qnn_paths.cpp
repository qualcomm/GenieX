// SPDX-License-Identifier: BSD-3-Clause
#include "internal/qnn_paths.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main() {
    namespace fs    = std::filesystem;
    namespace paths = geniex::qairt::paths;
    const auto root =
        fs::temp_directory_path() /
        ("geniex-tts-paths-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        // This test creates only known empty directories/files and removes them individually.
        fs::create_directories(root / "lib" / paths::kHostLibTriple);
        const auto library = root / "lib" / paths::kHostLibTriple / paths::kQnnBackendLib;
        std::ofstream(library).close();
        const auto host = paths::locate_qnn_host_lib_dir(root);
        if (host != library.parent_path()) throw std::runtime_error("SDK layout resolution");
        if (paths::locate_qnn_host_lib_dir(host) != host) throw std::runtime_error("flat layout resolution");
        if (!paths::locate_qnn_host_lib_dir(root / "missing").empty())
            throw std::runtime_error("missing runtime accepted");
        fs::create_directories(root / "lib/hexagon-v73/unsigned");
        fs::create_directories(root / "lib/hexagon-v75/unsigned");
        const auto adsp = paths::collect_adsp_library_path(root);
        if (paths::collect_adsp_library_path(host) != adsp) throw std::runtime_error("SDK host-only skel resolution");
        if (adsp.find(';') == std::string::npos || adsp.find("v73") == std::string::npos ||
            adsp.find("v75") == std::string::npos)
            throw std::runtime_error("FastRPC DSP path list");
        fs::remove(library);
        fs::remove(host);
        for (auto arch : {"hexagon-v73", "hexagon-v75"}) {
            fs::remove(root / "lib" / arch / "unsigned");
            fs::remove(root / "lib" / arch);
        }
        fs::remove(root / "lib");
        fs::remove(root);
        std::cout << "Shared QAIRT path contracts passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
