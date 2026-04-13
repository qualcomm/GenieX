#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <winsvc.h>

#include <vector>
#elif defined(__ANDROID__)
#include <dlfcn.h>
#endif

#include "logging.h"
#include "ml.h"

namespace geniex {

#if defined(_WIN32) || defined(__ANDROID__)
/**
 * @brief Detect HTP (Hexagon Tensor Processor) architecture version via CDSP.
 *
 * Dynamically loads libcdsprpc and queries the DSP for its architecture version
 * using the FastRPC remote_handle_control API.
 *
 * On Windows the DLL is located via the qcnspmcdm driver service (Qualcomm
 * driver store). On Android it is a system library on the default linker path.
 *
 * All constants are from the Qualcomm FastRPC public headers (BSD-3-Clause):
 *   https://github.com/qualcomm/fastrpc
 *
 * @return Detected arch version (e.g. 73, 75, 81), or 0 on failure.
 */
inline int detect_htp_arch() {
  static int s_arch = -1;
  if (s_arch >= 0)
    return s_arch;

  using remote_handle_control_t = int (*)(uint32_t, void *, uint32_t);
  remote_handle_control_t fn = nullptr;

#ifdef _WIN32
  // --- Windows: locate libcdsprpc.dll via the qcnspmcdm driver service ---
  SC_HANDLE scm = OpenSCManagerW(NULL, NULL, STANDARD_RIGHTS_READ);
  if (!scm) {
    GENIEX_LOG_WARN("HTP detect: cannot open SCManager ({})", GetLastError());
    s_arch = 0;
    return s_arch;
  }

  SC_HANDLE svc = OpenServiceW(scm, L"qcnspmcdm", SERVICE_QUERY_CONFIG);
  if (!svc) {
    GENIEX_LOG_WARN("HTP detect: qcnspmcdm service not found ({})",
                    GetLastError());
    CloseServiceHandle(scm);
    s_arch = 0;
    return s_arch;
  }

  DWORD buf_size = 0;
  QueryServiceConfigW(svc, NULL, 0, &buf_size);
  std::vector<uint8_t> cfg_buf(buf_size);
  auto *cfg = reinterpret_cast<LPQUERY_SERVICE_CONFIGW>(cfg_buf.data());
  if (!QueryServiceConfigW(svc, cfg, buf_size, &buf_size)) {
    GENIEX_LOG_WARN("HTP detect: QueryServiceConfigW failed ({})",
                    GetLastError());
    CloseServiceHandle(svc);
    CloseServiceHandle(scm);
    s_arch = 0;
    return s_arch;
  }

  std::wstring drv_dir(cfg->lpBinaryPathName);
  CloseServiceHandle(svc);
  CloseServiceHandle(scm);

  auto last_sep = drv_dir.find_last_of(L'\\');
  if (last_sep != std::wstring::npos)
    drv_dir.resize(last_sep);

  const std::wstring placeholder = L"\\SystemRoot";
  if (drv_dir.compare(0, placeholder.size(), placeholder) == 0) {
    wchar_t windir[MAX_PATH];
    if (GetEnvironmentVariableW(L"windir", windir, MAX_PATH))
      drv_dir.replace(0, placeholder.size(), windir);
  }

  std::wstring dll_path = drv_dir + L"\\libcdsprpc.dll";

  DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
  SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
  HMODULE lib = LoadLibraryW(dll_path.c_str());
  SetErrorMode(old_mode);

  if (!lib) {
    GENIEX_LOG_WARN(
        "HTP detect: failed to load libcdsprpc.dll from driver store");
    s_arch = 0;
    return s_arch;
  }

  fn = reinterpret_cast<remote_handle_control_t>(
      GetProcAddress(lib, "remote_handle_control"));
  if (!fn) {
    GENIEX_LOG_WARN(
        "HTP detect: remote_handle_control not found in libcdsprpc.dll");
    FreeLibrary(lib);
    s_arch = 0;
    return s_arch;
  }

#else // __ANDROID__
  // --- Android: libcdsprpc.so is a system library on the default path ---
  void *lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    GENIEX_LOG_WARN("HTP detect: failed to load libcdsprpc.so: {}", dlerror());
    s_arch = 0;
    return s_arch;
  }

  fn = reinterpret_cast<remote_handle_control_t>(
      dlsym(lib, "remote_handle_control"));
  if (!fn) {
    GENIEX_LOG_WARN(
        "HTP detect: remote_handle_control not found in libcdsprpc.so");
    dlclose(lib);
    s_arch = 0;
    return s_arch;
  }
#endif

  // FastRPC constants (from qualcomm/fastrpc public headers)
  constexpr uint32_t DSPRPC_GET_DSP_INFO = 2;
  constexpr uint32_t FASTRPC_ARCH_VER = 6;
  constexpr uint32_t FASTRPC_CDSP_DOMAIN = 3;

  struct {
    uint32_t domain;
    uint32_t attribute_ID;
    uint32_t capability;
  } cap{};
  cap.domain = FASTRPC_CDSP_DOMAIN;
  cap.attribute_ID = FASTRPC_ARCH_VER;

  int err = fn(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
  if (err != 0) {
    GENIEX_LOG_WARN("HTP detect: DSPRPC_GET_DSP_INFO failed (err={})", err);
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    s_arch = 0;
    return s_arch;
  }

  int arch = 0;
  switch (cap.capability & 0xff) {
  case 0x68:
    arch = 68;
    break;
  case 0x69:
    arch = 69;
    break;
  case 0x73:
    arch = 73;
    break;
  case 0x75:
    arch = 75;
    break;
  case 0x79:
    arch = 79;
    break;
  case 0x81:
    arch = 81;
    break;
  case 0x85:
    arch = 85;
    break;
  default:
    GENIEX_LOG_WARN("HTP detect: unknown arch capability 0x{:x}",
                    cap.capability);
#ifdef _WIN32
    FreeLibrary(lib);
#else
    dlclose(lib);
#endif
    s_arch = 0;
    return s_arch;
  }

  GENIEX_LOG_INFO("Detected HTP arch: v{}", arch);
  s_arch = arch;
  return s_arch;
}
#endif // _WIN32 || __ANDROID__

inline std::string fill_qnn_lib_path(std::string device_id = "",
                                     int htp_arch = 0) {
  std::string base_lib_path;

  auto plugin_path = std::getenv("GENIEX_PLUGIN_PATH");
  if (plugin_path) {
    base_lib_path = (std::filesystem::path(plugin_path) /
                     GENIEX_PLUGIN_ID_QAIRT / "htp-files")
                        .string();
  }

#ifdef __ANDROID__
  // On Android, HTP runtime libraries are extracted from APK assets into
  // app-internal storage. GENIEX_QNN_HTP_PATH points to that directory
  // (e.g. /data/user/0/<pkg>/files/npu/) which contains htp-files[-vXX]/.
  if (base_lib_path.empty() || !std::filesystem::exists(base_lib_path)) {
    auto htp_path = std::getenv("GENIEX_QNN_HTP_PATH");
    if (htp_path) {
      base_lib_path = (std::filesystem::path(htp_path) / "htp-files").string();
      GENIEX_LOG_DEBUG("Using GENIEX_QNN_HTP_PATH for HTP libs: {}",
                       base_lib_path);
    }
  }
#endif

  if (base_lib_path.empty()) {
    GENIEX_LOG_WARN("No HTP lib path found (GENIEX_PLUGIN_PATH and "
                    "GENIEX_QNN_HTP_PATH are unset or invalid).");
    return "";
  }

  std::string lib_path = base_lib_path;

#if defined(_WIN32) || defined(__ANDROID__)
  auto try_versioned_path = [&](int version) {
    auto versioned = base_lib_path + "-v" + std::to_string(version);
    if (std::filesystem::exists(versioned)) {
      lib_path = versioned;
      GENIEX_LOG_INFO("Using v{} HTP lib path: {}", version, lib_path);
      return true;
    }
    GENIEX_LOG_DEBUG("v{} path not found at {}, falling back to: {}", version,
                     versioned, base_lib_path);
    return false;
  };

  if (htp_arch == 85) {
    try_versioned_path(85);
  } else if (htp_arch == 81) {
    try_versioned_path(81);
  }
#elif defined(__linux__) && !defined(__ANDROID__)
  if (device_id.empty() || device_id == "iq9")
    lib_path += "-iq9-v79-linux-gcc11.2";
  else if (device_id == "rb3")
    lib_path += "-rb3-v68-linux-gcc11.2";
#endif

  GENIEX_LOG_DEBUG("Determined lib path: {}", lib_path);

#ifdef _WIN32
  _putenv_s("ADSP_LIBRARY_PATH", lib_path.c_str());
#else
  setenv("ADSP_LIBRARY_PATH", lib_path.c_str(), 1);
#endif
  GENIEX_LOG_DEBUG("ADSP_LIBRARY_PATH set to {}", lib_path);

  return lib_path;
}

/**
 * @brief RAII helper to inject QNN library and model folder paths from config.
 *
 * Auto-fills qnn_model_folder_path and qnn_lib_folder_path if not provided.
 * When a preset_lib_path is supplied (computed once at plugin init), it is used
 * directly instead of re-computing from GENIEX_PLUGIN_PATH, avoiding redundant
 * work and ensuring all models share the same arch-aware path.
 *
 * On Android AAR apps, preset_lib_path always takes priority over the
 * Java-provided qnn_lib_folder_path, because Java defaults to nativeLibraryDir
 * which is flat and doesn't contain the arch-specific HTP runtime libraries
 * (they're extracted from APK assets into a separate directory with proper
 * structure).
 *
 * Model path is derived from the input model_path. Library path is taken from
 * preset_lib_path if non-empty, otherwise resolved from GENIEX_PLUGIN_PATH env
 * var.
 */
template <typename T> class QnnFolderPathFiller {
private:
  std::string model_path, lib_path;

  void fill_lib_path(T *mutable_value, const std::string &preset_lib_path) {
#ifdef __ANDROID__
    // On Android AAR apps, Java sets qnn_lib_folder_path to nativeLibraryDir
    // which is flat and doesn't contain arch-specific HTP runtime libraries.
    // Always prefer the arch-aware preset_lib_path from compute_lib_path().
    if (!preset_lib_path.empty()) {
      lib_path = preset_lib_path;
      GENIEX_LOG_DEBUG("Using preset lib path: {}", lib_path);
      mutable_value->config.qnn_lib_folder_path = lib_path.c_str();
      return;
    }
#endif
    if (!mutable_value->config.qnn_lib_folder_path) {
      if (!preset_lib_path.empty()) {
        lib_path = preset_lib_path;
        GENIEX_LOG_DEBUG("Using preset lib path: {}", lib_path);
      } else {
        lib_path = fill_qnn_lib_path(
            mutable_value->device_id ? mutable_value->device_id : "");
      }
      mutable_value->config.qnn_lib_folder_path = lib_path.c_str();
    }
  }

public:
  QnnFolderPathFiller(const T *value, const std::string &preset_lib_path = "") {
    auto *mutable_value = const_cast<T *>(value);

    if (!mutable_value->config.qnn_model_folder_path) {
      model_path =
          std::filesystem::path(value->model_path).parent_path().string();
      mutable_value->config.qnn_model_folder_path = model_path.c_str();
    }

    fill_lib_path(mutable_value, preset_lib_path);

    GENIEX_LOG_DEBUG("fill npu folder path in input: {}", value);
  }
};

/**
 * @brief Specialized QnnFolderPathFiller for CV models.
 *
 * Uses det_model_path instead of model_path for determining the model folder.
 */
template <> class QnnFolderPathFiller<ml_CVCreateInput> {
private:
  std::string model_path, lib_path;

public:
  QnnFolderPathFiller(const ml_CVCreateInput *value,
                      const std::string &preset_lib_path = "") {
    auto *mutable_value = const_cast<ml_CVCreateInput *>(value);

    if (!mutable_value->config.qnn_model_folder_path) {
      model_path = std::filesystem::path(value->config.det_model_path)
                       .parent_path()
                       .string();
      mutable_value->config.qnn_model_folder_path = model_path.c_str();
    }

#ifdef __ANDROID__
    if (!preset_lib_path.empty()) {
      lib_path = preset_lib_path;
      GENIEX_LOG_DEBUG("Using preset lib path: {}", lib_path);
      mutable_value->config.qnn_lib_folder_path = lib_path.c_str();
    } else
#endif
        if (!mutable_value->config.qnn_lib_folder_path) {
      if (!preset_lib_path.empty()) {
        lib_path = preset_lib_path;
        GENIEX_LOG_DEBUG("Using preset lib path: {}", lib_path);
      } else {
        lib_path = fill_qnn_lib_path(
            mutable_value->device_id ? mutable_value->device_id : "");
      }
      mutable_value->config.qnn_lib_folder_path = lib_path.c_str();
    }

    GENIEX_LOG_DEBUG("fill npu folder path in input: {}", value);
  }
};

} // namespace geniex
