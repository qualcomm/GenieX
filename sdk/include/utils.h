#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "geniex.h"

namespace geniex {

/**
 * @brief Get the directory containing the shared library (geniex_bridge)
 *
 * On Windows, uses GetModuleFileName to find the DLL path.
 * On Unix systems, uses dladdr to find the shared object path.
 *
 * @return Filesystem path to the directory containing the shared library
 * @throws std::runtime_error if the directory cannot be determined (Unix only)
 */
std::filesystem::path get_shared_lib_dir();

/**
 * @brief Helper function to process a buffer and extract only valid UTF-8 string portions
 *
 * This function handles incomplete UTF-8 sequences (like emojis split across token boundaries).
 * It extracts complete UTF-8 characters from the buffer and leaves incomplete sequences for
 * the next call.
 *
 * @param buffer Reference to the buffer containing potentially incomplete UTF-8 data.
 *               Complete UTF-8 sequences will be removed from the buffer.
 * @return String containing only complete UTF-8 sequences ready to be sent
 */
std::string valid_utf8(std::string& buffer);

/**
 * @brief Wrapper structure for UTF-8 validation callbacks
 *
 * This struct wraps user callbacks to accumulate incomplete UTF-8 sequences
 * and only pass complete UTF-8 characters to the original callback.
 */
struct Utf8CallbackWrapper {
    geniex_token_callback original_callback;
    void*                 original_user_data;
    std::string           utf8_buffer;

    /**
     * @brief Flush any remaining incomplete UTF-8 sequence
     *
     * Should be called after generation completes to send any remaining buffered data.
     */
    void flush() {
        if (!utf8_buffer.empty() && original_callback) {
            original_callback(utf8_buffer.c_str(), original_user_data);
        }
    }
};

/**
 * @brief Get the UTF-8 validation callback wrapper function
 *
 * Returns a callback function that wraps user callbacks to handle incomplete UTF-8 sequences.
 * The wrapper accumulates incomplete sequences and only passes complete UTF-8 characters
 * to the original callback.
 *
 * @return Function pointer suitable for use as geniex_token_callback
 */
geniex_token_callback get_utf8_callback_wrapper();

/**
 * @brief Replace all occurrences of a substring in a string
 *
 * @param str The string to modify (modified in place)
 * @param from The substring to search for
 * @param to The replacement string
 */
inline void string_replace_all(std::string& str, const std::string& from, const std::string& to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
}

}  // namespace geniex
