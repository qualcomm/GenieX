#include "utils.h"

#if defined(_WIN32)
#include <windows.h>
extern "C" IMAGE_DOS_HEADER __ImageBase;
#else
#include <dlfcn.h>
#endif

namespace geniex {

std::filesystem::path get_shared_lib_dir() {
#if defined(_WIN32)
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
#else
    // Use dladdr to get the path of the dynamic library containing this function
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&get_shared_lib_dir), &info) && info.dli_fname) {
        return std::filesystem::canonical(info.dli_fname).parent_path();
    }
    throw std::runtime_error("Cannot determine shared library directory");
#endif
}

std::string valid_utf8(std::string& buffer) {
    if (buffer.empty()) return "";

    // Find the start of the last potential character in the buffer
    size_t last_char_start = buffer.length();
    while (last_char_start > 0) {
        last_char_start--;
        unsigned char c = buffer[last_char_start];
        // Stop if a byte that is NOT a continuation byte (it's a start byte or ASCII)
        if ((c & 0xC0) != 0x80) {
            break;
        }
    }

    // Compute the expected length of the character starting at that position
    size_t expected_char_len = 0;
    if (last_char_start < buffer.length()) {
        unsigned char first_byte = buffer[last_char_start];
        if ((first_byte & 0x80) == 0x00)
            expected_char_len = 1;  // ASCII
        else if ((first_byte & 0xE0) == 0xC0)
            expected_char_len = 2;  // 2-byte
        else if ((first_byte & 0xF0) == 0xE0)
            expected_char_len = 3;  // 3-byte
        else if ((first_byte & 0xF8) == 0xF0)
            expected_char_len = 4;  // 4-byte
        else {
            // Invalid byte sequence, treat as a single byte to advance past it
            expected_char_len = 1;
        }
    }

    size_t      bytes_in_last_char = buffer.length() - last_char_start;
    std::string string_to_send;

    if (expected_char_len > 0 && bytes_in_last_char >= expected_char_len) {
        // all bytes for last character in buffer are there. Send the whole buffer.
        string_to_send = buffer;
        buffer.clear();
    } else {
        // The last character is incomplete. Send everything *before* it.
        string_to_send = buffer.substr(0, last_char_start);
        buffer.erase(0, last_char_start);
    }

    return string_to_send;
}

geniex_token_callback get_utf8_callback_wrapper() {
    return [](const char* token, void* user_data) -> bool {
        auto* wrapper_data = static_cast<Utf8CallbackWrapper*>(user_data);
        wrapper_data->utf8_buffer.append(token);
        std::string valid_part = geniex::valid_utf8(wrapper_data->utf8_buffer);
        if (valid_part.length() > 0 && wrapper_data->original_callback) {
            return wrapper_data->original_callback(valid_part.c_str(), wrapper_data->original_user_data);
        }
        return true;
    };
}

}  // namespace geniex
