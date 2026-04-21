#pragma once

/**
 * @file validation.h
 * @brief Header-only license validation utilities
 *
 * This header contains all license validation functionality including offline/online
 * validation, activation, and license management in a single header-only implementation.
 */

#ifdef _WIN32
// Ensure winsock2 is included before windows.h to avoid conflicts with httplib/OpenSSL
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include <openssl/bio.h>
#include <openssl/pem.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Wbemidl.h>
#include <comdef.h>
#include <windows.h>
#pragma comment(lib, "wbemuuid.lib")
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

#include "external/date.h"
#include "external/httplib.h"
#include "external/json.hpp"
#include "geniex.h"

namespace geniex {
namespace validation {

// =============================================================================
// Constants
// =============================================================================

#define LICENSE_KEY_HEAD "key/"

// public key for offline key validation, for more details, refer to https://keygen.sh/docs/api/cryptography/
#define ENC_PUBLIC_KEY "1a7a8acc8cf8597a2cfae03c9151dab85a28f3a4b40b07bcd4fbb6101b7f5178"

// keygen server endpoint for online key validation
#define LICENSE_SERVER_ENDPOINT "https://lic.geniex.ai"

// delimiter for separating license components in storage file
#define LICENSE_DELIMITER "<:>"

// =============================================================================
// Data Structures
// =============================================================================

struct ValidationResult {
    // ML standard error codes for validation results
    geniex_ErrorCode
        result;  // GENIEX_SUCCESS, GENIEX_ERROR_COMMON_LICENSE_INVALID, GENIEX_ERROR_COMMON_LICENSE_EXPIRED, etc.
    std::string message;
};

// =============================================================================
// Utility Functions
// =============================================================================

inline std::string resolve_license_key(const std::string& provided_key) {
    // If license key is provided and not empty, use it
    if (!provided_key.empty()) {
        return provided_key;
    }

    // Otherwise, try to get from GENIEX_TOKEN environment variable
    const char* env_token = std::getenv("GENIEX_TOKEN");
    if (env_token && strlen(env_token) > 0) {
        return std::string(env_token);
    }

    // Return empty string if neither is available
    return "";
}

inline std::vector<unsigned char> hex_to_bytes(const std::string& hex) {
    if (hex.size() % 2 != 0) {
        return std::vector<unsigned char>();
    }

    std::vector<unsigned char> bytes(hex.size() / 2);
    for (size_t i = 0; i < bytes.size(); ++i) {
        unsigned int      byte;
        std::stringstream ss;
        ss << std::hex << hex.substr(i * 2, 2);
        ss >> byte;
        bytes[i] = static_cast<unsigned char>(byte);
    }
    return bytes;
}

inline std::vector<unsigned char> base64_urlsafe_decode(const std::string& input) {
    std::string b64 = input;
    std::replace(b64.begin(), b64.end(), '-', '+');
    std::replace(b64.begin(), b64.end(), '_', '/');
    while (b64.size() % 4) {
        b64 += '=';
    }

    BIO* bio    = BIO_new_mem_buf(b64.data(), b64.size());
    BIO* b64bio = BIO_new(BIO_f_base64());
    BIO_set_flags(b64bio, BIO_FLAGS_BASE64_NO_NL);
    bio = BIO_push(b64bio, bio);

    std::vector<unsigned char> out(b64.size());
    int                        decoded_len = BIO_read(bio, out.data(), b64.size());
    if (decoded_len <= 0) {
        return std::vector<unsigned char>();
    }

    out.resize(decoded_len);
    BIO_free_all(bio);
    return out;
}

inline time_t convertTime(const std::string& s) {
    std::istringstream in(s);
    in.exceptions(std::ios::failbit);

    date::sys_time<std::chrono::milliseconds> tp;
    in >> date::parse("%FT%TZ", tp);

    return tp.time_since_epoch().count() / 1000;
}

// =============================================================================
// Core Validation Functions
// =============================================================================

inline ValidationResult validate_license_key_offline(const std::string& provided_key = "") {
    // Resolve the license key to use (parameter or environment variable)
    std::string license_key = resolve_license_key(provided_key);
    if (license_key.empty()) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
            "No license key provided and GENIEX_TOKEN environment variable not set"};
    }

    if (license_key.substr(0, 4) != LICENSE_KEY_HEAD) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "invalid license key format: missing key/ prefix"};
    }

    auto        pos          = license_key.rfind(".");
    std::string data_part    = license_key.substr(0, pos);
    std::string sig_b64_part = license_key.substr(pos + 1);

    auto data      = base64_urlsafe_decode(data_part.substr(4));
    auto signature = base64_urlsafe_decode(sig_b64_part);

    auto pkey_bytes = hex_to_bytes(ENC_PUBLIC_KEY);

    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, pkey_bytes.data(), pkey_bytes.size());
    if (!pkey) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to decode ED25519 public key"};
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to new signature verification context"};
    } else if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) != 1) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to initialize signature verification"};
    }

    auto ret = EVP_DigestVerify(ctx,
        signature.data(),
        signature.size(),
        reinterpret_cast<const unsigned char*>(data_part.data()),
        data_part.size());

    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);

    if (ret != 1) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "signature verification failed"};
    }

    try {
        auto detail = nlohmann::json::parse(std::string(data.begin(), data.end()));
        if (!detail.contains("license") || !detail["license"].is_object() || !detail["license"].contains("expiry") ||
            !detail["license"]["expiry"].is_string() || !detail["license"].contains("created") ||
            !detail["license"]["created"].is_string()) {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "invalid license data format"};
        }

        auto created_at = convertTime(detail["license"]["created"].get<std::string>());
        auto expired_at = convertTime(detail["license"]["expiry"].get<std::string>());
        auto now        = std::time(nullptr);
        if (now < created_at || now > expired_at) {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_EXPIRED, "license expired or not yet valid"};
        }
        return ValidationResult{GENIEX_SUCCESS, "verification success"};
    } catch (nlohmann::json::parse_error& e) {
        return ValidationResult{
            GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to parse license data: " + std::string(e.what())};
    }
}

inline ValidationResult validate_license_key_online(const std::string& provided_key = "") {
    // Resolve the license key to use (parameter or environment variable)
    std::string license_key = resolve_license_key(provided_key);
    if (license_key.empty()) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
            "No license key provided and GENIEX_TOKEN environment variable not set"};
    }

    httplib::Client cli(LICENSE_SERVER_ENDPOINT);

    nlohmann::json body;
    body["meta"]["key"] = license_key;
    auto resp           = cli.Post("/v1/licenses/actions/validate-key", body.dump(), "application/json");
    if (!resp || resp->status != 200) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to request license host"};
    }

    try {
        nlohmann::json pj = nlohmann::json::parse(resp->body);
        if (!pj.contains("meta") || !pj["meta"].is_object() || !pj["meta"].contains("code") ||
            !pj["meta"]["code"].is_string()) {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "invalid validation reponse format"};
        }

        // refer to https://keygen.sh/docs/api/licenses/#licenses-actions-validate-key
        auto code = pj["meta"]["code"].get<std::string>();

        if (code == "VALID") {
            return ValidationResult{GENIEX_SUCCESS, "verification success"};
        } else if (code == "EXPIRED") {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_EXPIRED, "license expired: " + code};
        } else {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "invalid license status: " + code};
        }
    } catch (nlohmann::json::parse_error& e) {
        return ValidationResult{
            GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to parse validation response: " + std::string(e.what())};
    }
}

// =============================================================================
// System Information Functions
// =============================================================================

inline std::string get_device_id() {
    std::string device_id;

#ifdef _WIN32
    // Try to get Windows system UUID using WMI
    HRESULT hres;

    // Initialize COM
    hres = CoInitializeEx(0, COINIT_MULTITHREADED);
    if (SUCCEEDED(hres)) {
        // Set general COM security
        hres = CoInitializeSecurity(
            NULL, -1, NULL, NULL, RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE, NULL, EOAC_NONE, NULL);

        if (SUCCEEDED(hres)) {
            // Obtain WMI locator
            IWbemLocator* pLoc = NULL;
            hres = CoCreateInstance(CLSID_WbemLocator, 0, CLSCTX_INPROC_SERVER, IID_IWbemLocator, (LPVOID*)&pLoc);

            if (SUCCEEDED(hres)) {
                // Connect to WMI namespace ROOT\CIMV2
                IWbemServices* pSvc = NULL;
                hres                = pLoc->ConnectServer(_bstr_t(L"ROOT\\CIMV2"),  // WMI namespace
                    NULL,
                    NULL,
                    0,
                    NULL,
                    0,
                    0,
                    &pSvc);

                if (SUCCEEDED(hres)) {
                    // Set proxy blanket
                    hres = CoSetProxyBlanket(pSvc,
                        RPC_C_AUTHN_WINNT,
                        RPC_C_AUTHZ_NONE,
                        NULL,
                        RPC_C_AUTHN_LEVEL_CALL,
                        RPC_C_IMP_LEVEL_IMPERSONATE,
                        NULL,
                        EOAC_NONE);

                    if (SUCCEEDED(hres)) {
                        // Execute WMI query
                        IEnumWbemClassObject* pEnumerator = NULL;
                        hres                              = pSvc->ExecQuery(bstr_t("WQL"),
                            bstr_t("SELECT UUID FROM Win32_ComputerSystemProduct"),
                            WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                            NULL,
                            &pEnumerator);

                        if (SUCCEEDED(hres)) {
                            // Get result
                            IWbemClassObject* pclsObj = NULL;
                            ULONG             uReturn = 0;

                            HRESULT hr = pEnumerator->Next(WBEM_INFINITE, 1, &pclsObj, &uReturn);
                            if (uReturn != 0) {
                                VARIANT vtProp;
                                hr = pclsObj->Get(L"UUID", 0, &vtProp, 0, 0);
                                if (SUCCEEDED(hr) && vtProp.bstrVal != NULL) {
                                    // Convert wide string to regular string
                                    int len = WideCharToMultiByte(CP_UTF8, 0, vtProp.bstrVal, -1, NULL, 0, NULL, NULL);
                                    if (len > 0) {
                                        char* buffer = new char[len];
                                        WideCharToMultiByte(CP_UTF8, 0, vtProp.bstrVal, -1, buffer, len, NULL, NULL);
                                        device_id = std::string(buffer);
                                        delete[] buffer;
                                    }
                                }
                                VariantClear(&vtProp);
                                pclsObj->Release();
                            }
                            pEnumerator->Release();
                        }
                    }
                    pSvc->Release();
                }
                pLoc->Release();
            }
        }
        CoUninitialize();
    }

    // If WMI failed, try to get computer name as fallback
    if (device_id.empty()) {
        char  computer_name[MAX_COMPUTERNAME_LENGTH + 1];
        DWORD size = sizeof(computer_name);
        if (GetComputerNameA(computer_name, &size)) {
            device_id = std::string(computer_name);
        }
    }

#else
    // Try to read /etc/machine-id on Linux
    std::ifstream machine_id_file("/etc/machine-id");
    if (machine_id_file.is_open()) {
        std::getline(machine_id_file, device_id);
        machine_id_file.close();

        // Remove any whitespace
        device_id.erase(std::remove_if(device_id.begin(), device_id.end(), ::isspace), device_id.end());
    }

    // Fallback to hostname if machine-id not available
    if (device_id.empty()) {
        struct utsname system_info;
        if (uname(&system_info) == 0) {
            device_id = std::string(system_info.nodename);
        }
    }

    // Final fallback to gethostname
    if (device_id.empty()) {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) == 0) {
            device_id = std::string(hostname);
        }
    }
#endif

    // If all methods failed, generate a basic fallback
    if (device_id.empty()) {
        device_id = "unknown-device";
    }

    return device_id;
}

inline std::string get_license_file_path() {
    // Get home directory path (cross-platform)
    std::string home_dir;

#ifdef _WIN32
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) {
        home_dir = userprofile;
    } else {
        const char* homedrive = std::getenv("HOMEDRIVE");
        const char* homepath  = std::getenv("HOMEPATH");
        if (homedrive && homepath) {
            home_dir = std::string(homedrive) + homepath;
        }
    }
#else
    const char* home = std::getenv("HOME");
    if (home) {
        home_dir = home;
    }
#endif

    if (home_dir.empty()) {
        throw std::runtime_error("Unable to determine home directory");
    }

    // Create the .geniex directory path
    std::filesystem::path geniex_dir = std::filesystem::path(home_dir) / ".geniex";

    // Create the directory if it doesn't exist
    try {
        std::filesystem::create_directories(geniex_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        // If we can't create the directory, fallback to home directory
        geniex_dir = home_dir;
    }

    // Create the license tracking file path
    std::filesystem::path license_file_path = geniex_dir / "licenses.txt";

    return license_file_path.string();
}

// =============================================================================
// License Management Functions
// =============================================================================

inline std::string decode_license_id_from_key(const std::string& provided_key = "") {
    try {
        // Resolve the license key to use (parameter or environment variable)
        std::string license_key = resolve_license_key(provided_key);
        if (license_key.empty()) {
            return "";  // No license key available
        }

        // Check if license key has the correct format
        if (license_key.substr(0, 4) != LICENSE_KEY_HEAD) {
            return "";  // Invalid format
        }

        // Find the last dot to separate data from signature
        auto pos = license_key.rfind(".");
        if (pos == std::string::npos) {
            return "";  // Invalid format
        }

        // Extract the data part (before the signature)
        std::string data_part = license_key.substr(0, pos);

        // Decode the base64 data (skip the "key/" prefix)
        auto data = base64_urlsafe_decode(data_part.substr(4));
        if (data.empty()) {
            return "";  // Failed to decode
        }

        // Parse the JSON data
        nlohmann::json detail;
        try {
            detail = nlohmann::json::parse(std::string(data.begin(), data.end()));
        } catch (const std::exception& e) {
            return "";
        }

        // Extract license ID
        if (detail.contains("license") && detail["license"].is_object() && detail["license"].contains("id") &&
            detail["license"]["id"].is_string()) {
            return detail["license"]["id"].get<std::string>();
        }

        return "";  // License ID not found or invalid format
    } catch (const std::exception& e) {
        return "";  // Return empty string on any error
    }
}

inline ValidationResult activate(const std::string& provided_key = "", const std::string& provided_id = "") {
    // Resolve the license key to use (parameter or environment variable)
    std::string license_key = resolve_license_key(provided_key);
    if (license_key.empty()) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
            "No license key provided and GENIEX_TOKEN environment variable not set"};
    }

    httplib::Client cli(LICENSE_SERVER_ENDPOINT);

    // Determine the license ID to use
    std::string license_id = provided_id;
    if (license_id.empty()) {
        // If license_id is not provided, decode it from the license key
        license_id = decode_license_id_from_key(license_key);
        if (license_id.empty()) {
            return ValidationResult{
                GENIEX_ERROR_COMMON_LICENSE_INVALID, "failed to decode license ID from license key"};
        }
    }

    // Create the JSON body structure as shown in the curl example
    nlohmann::json body;
    body["data"]["type"]                      = "machines";
    body["data"]["attributes"]["fingerprint"] = get_device_id();  // Use actual device ID
#ifdef _WIN32
    body["data"]["attributes"]["platform"] = "Windows";
#else
    body["data"]["attributes"]["platform"] = "Linux";
#endif
    body["data"]["relationships"]["license"]["data"]["type"] = "licenses";
    body["data"]["relationships"]["license"]["data"]["id"]   = license_id;

    // Set up headers
    httplib::Headers headers = {{"Content-Type", "application/vnd.api+json"},
        {"Accept", "application/vnd.api+json"},
        {"Authorization", "License " + license_key}};

    auto resp = cli.Post("/v1/machines", headers, body.dump(), "application/vnd.api+json");
    if (!resp) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "fail to request license host for activation"};
    }

    if (resp->status == 200 || resp->status == 201) {
        return ValidationResult{GENIEX_SUCCESS, "activation success"};
    } else if (resp->status == 422) {
        // Handle the specific case where fingerprint is already taken
        try {
            nlohmann::json error_response = nlohmann::json::parse(resp->body);

            // Check if errors array exists and has exactly one element
            if (error_response.contains("errors") && error_response["errors"].is_array() &&
                error_response["errors"].size() == 1 && error_response["errors"][0].contains("code") &&
                error_response["errors"][0]["code"].is_string() &&
                error_response["errors"][0]["code"].get<std::string>() == "FINGERPRINT_TAKEN") {
                return ValidationResult{GENIEX_SUCCESS, "activation success (fingerprint already registered)"};
            }
        } catch (const nlohmann::json::parse_error& e) {
            // If JSON parsing fails, fall through to the general error case
        }

        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
            "activation failed with status: " + std::to_string(resp->status) + ", response: " + resp->body};
    } else {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
            "activation failed with status: " + std::to_string(resp->status) + ", response: " + resp->body};
    }
}

inline bool is_first_time_license(const std::string& provided_key = "", const std::string& /* provided_id */ = "") {
    // Resolve the license key to use (parameter or environment variable)
    std::string license_key = resolve_license_key(provided_key);
    if (license_key.empty()) {
        return true;  // If no license key, treat as first time
    }

    // Get the license file path using the helper function
    std::string license_file_path = get_license_file_path();

    // Create identifier from license key only (ignoring license_id)
    std::string license_identifier = license_key;

    // Read existing license keys from file
    std::ifstream infile(license_file_path);
    if (infile.is_open()) {
        std::string line;
        while (std::getline(infile, line)) {
            // Check if line starts with license_identifier + delimiter (to handle timestamp format)
            // Line format is now: license_key + delimiter + timestamp
            std::string delimiter_check = license_identifier + LICENSE_DELIMITER;
            if (line.length() > delimiter_check.length() &&
                line.substr(0, delimiter_check.length()) == delimiter_check) {
                infile.close();
                return false;  // License key already exists, not first time
            }
        }
        infile.close();
    }

    return true;  // License key not found, this is the first time
}

inline void record_license(const std::string& provided_key = "", const std::string& /* provided_id */ = "") {
    // Resolve the license key to use (parameter or environment variable)
    std::string license_key = resolve_license_key(provided_key);
    if (license_key.empty()) {
        return;  // If no license key, nothing to record
    }

    // Get the license file path using the helper function
    std::string license_file_path = get_license_file_path();

    // Get current unix timestamp
    auto now       = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    // Create identifier with timestamp using license key only (ignoring license_id)
    std::string license_identifier = license_key + LICENSE_DELIMITER + std::to_string(timestamp);

    // Append the license identifier to the file
    std::ofstream outfile(license_file_path, std::ios::app);
    if (outfile.is_open()) {
        outfile << license_identifier << std::endl;
        outfile.close();
    } else {
        throw std::runtime_error("Unable to write to license file");
    }
}

// =============================================================================
// Main Validation Function
// =============================================================================

inline ValidationResult validate_license(const std::string& provided_key = "", const std::string& provided_id = "") {
    try {
        // Resolve the license key to use (parameter or environment variable)
        std::string license_key = resolve_license_key(provided_key);
        if (license_key.empty()) {
            return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID,
                "No license key provided and GENIEX_TOKEN environment variable not set"};
        }

        // Step 1: Determine if this license key is seen for the first time (ignoring license_id)
        bool is_first_time = is_first_time_license(license_key, provided_id);

        if (is_first_time) {
            // Step 2a: For first time, activate the license first
            ValidationResult activation_result = activate(license_key, provided_id);
            if (activation_result.result != GENIEX_SUCCESS) {
                return activation_result;  // Return activation failure
            }

            // Step 2b: After successful activation, validate the license online
            ValidationResult online_validation = validate_license_key_online(license_key);
            if (online_validation.result != GENIEX_SUCCESS) {
                return online_validation;  // Return online validation failure
            }

            // Step 2c: Only if both activation and validation succeeded, mark the license as seen
            record_license(license_key, provided_id);

            return ValidationResult{GENIEX_SUCCESS, "first time activation and validation success"};
        } else {
            // Step 3: For subsequent times, use offline validation
            return validate_license_key_offline(license_key);
        }
    } catch (const std::exception& e) {
        return ValidationResult{GENIEX_ERROR_COMMON_LICENSE_INVALID, "validation failed: " + std::string(e.what())};
    }
}

#undef LICENSE_KEY_HEAD
#undef ENC_PUBLIC_KEY
#undef LICENSE_SERVER_ENDPOINT
#undef LICENSE_DELIMITER

}  // namespace validation
}  // namespace geniex
