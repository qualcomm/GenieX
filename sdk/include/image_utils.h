#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "logging.h"

namespace geniex {
namespace image_utils {

/**
 * @brief Generate a temporary filename with random suffix
 */
inline std::string generate_temp_filename(const std::string& prefix, const std::string& ext) {
    static std::mt19937                       rng(std::random_device{}());
    static std::uniform_int_distribution<int> dist(0, 0xFFFFFF);

    std::ostringstream oss;
    oss << prefix << std::hex << std::setw(6) << std::setfill('0') << dist(rng) << ext;
    return (std::filesystem::temp_directory_path() / std::filesystem::path(oss.str())).string();
}

/**
 * @brief Get image dimensions using ffprobe
 */
inline std::pair<int, int> get_image_dimensions(const char* image_path) {
    auto dimfile = generate_temp_filename("image-dims-", ".txt");

    // Get image dimensions using ffprobe output to file
    std::string probe_cmd =
        "ffprobe -v quiet -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"" +
        std::string(image_path) + "\" > \"" + dimfile + "\"";

    int probe_ret = std::system(probe_cmd.c_str());
    if (probe_ret != 0) {
        throw std::runtime_error("ffprobe failed to get image dimensions");
    }

    // Read dimensions from file
    std::ifstream dim_file(dimfile);
    if (!dim_file.is_open()) {
        throw std::runtime_error("Failed to read image dimensions file");
    }

    std::string result;
    std::getline(dim_file, result);
    dim_file.close();

    // Clean up temp file
    std::remove(dimfile.c_str());

    // Parse dimensions (format: widthxheight)
    size_t x_pos = result.find('x');
    if (x_pos == std::string::npos) {
        throw std::runtime_error("Failed to parse image dimensions: " + result);
    }

    int width  = std::stoi(result.substr(0, x_pos));
    int height = std::stoi(result.substr(x_pos + 1));

    return std::make_pair(width, height);
}

/**
 * @brief Resize and pad image to square dimensions with intelligent aspect ratio handling
 *
 * This function processes images based on their aspect ratio:
 * - When aspect ratio is between 0.5-2.0: Directly resize to target size (may distort image slightly)
 * - When aspect ratio is outside 0.5-2.0: Resize maintaining aspect ratio and pad shorter sides
 * - Automatically detects edge color from top-left corner for padding background
 * - Falls back to direct resize if dimension detection fails
 *
 * @param image_path Path to the input image file
 * @param size Target square dimensions (default: 448x448)
 * @return Path to the processed temporary image file
 * @throws std::runtime_error if ffmpeg operations fail
 */
inline std::string resize_and_pad_image(const char* image_path, int size = 448) {
    auto outfile = generate_temp_filename("resized-", ".jpg");

    try {
        // Step 1: Get original image dimensions using ffprobe
        auto [width, height] = get_image_dimensions(image_path);
        double aspect_ratio  = static_cast<double>(width) / height;

        // Step 2: For normal aspect ratios (0.5-2.0), directly resize without padding
        // This handles most common image formats (landscape, portrait, square) efficiently
        if (aspect_ratio >= 0.5 && aspect_ratio <= 2.0) {
            std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) +
                              "\" -vf \"scale=" + std::to_string(size) + ":" + std::to_string(size) + "\" \"" +
                              outfile + "\"";

            int ret = std::system(cmd.c_str());
            if (ret != 0) {
                throw std::runtime_error("ffmpeg resize failed: " + cmd);
            }

            GENIEX_LOG_DEBUG("resize image completed successfully, output: {}", fmt::to_string(outfile));
            return outfile;
        }
    } catch (const std::exception&) {
        // Step 3: Fallback - If dimension detection fails, directly resize to target size
        std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) +
                          "\" -vf \"scale=" + std::to_string(size) + ":" + std::to_string(size) + "\" \"" + outfile +
                          "\"";

        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            throw std::runtime_error("ffmpeg resize failed: " + cmd);
        }

        GENIEX_LOG_DEBUG("resize image (fallback) completed successfully, output: {}", fmt::to_string(outfile));
        return outfile;
    }

    // Step 4: For extreme aspect ratios (< 0.5 or > 2.0), use smart padding
    // This preserves image quality for very wide or very tall images

    // Step 4a: Auto-detect padding color by sampling top-left corner pixel
    auto        color_file = generate_temp_filename("edge-color-", ".rgb");
    std::string sample_cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) +
                             "\" -vf \"crop=1:1:0:0\" -f rawvideo -pix_fmt rgb24 \"" + color_file + "\"";

    std::string bg_color = "0x808080";  // neutral gray fallback color

    // Try to extract edge color from top-left corner (1x1 pixel sample)
    if (std::system(sample_cmd.c_str()) == 0) {
        std::ifstream file(color_file, std::ios::binary);
        if (file.is_open()) {
            unsigned char rgb[3];
            if (file.read(reinterpret_cast<char*>(rgb), 3)) {
                // Convert RGB values to hex color format for ffmpeg
                std::ostringstream color_stream;
                color_stream << "0x" << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(rgb[0])
                             << std::setw(2) << static_cast<int>(rgb[1]) << std::setw(2) << static_cast<int>(rgb[2]);
                bg_color = color_stream.str();
            }
            file.close();
        }
    }

    // Clean up temporary color detection file
    std::remove(color_file.c_str());

    // Step 4b: Resize maintaining aspect ratio and pad shorter sides with detected color
    // force_original_aspect_ratio=decrease ensures the image fits within target size
    // pad centers the image and fills empty space with the detected background color
    std::string cmd = "ffmpeg -hide_banner -loglevel error -y -i \"" + std::string(image_path) + "\" " +
                      "-vf \"scale=" + std::to_string(size) + ":" + std::to_string(size) +
                      ":force_original_aspect_ratio=decrease," + "pad=" + std::to_string(size) + ":" +
                      std::to_string(size) + ":(ow-iw)/2:(oh-ih)/2:color=" + bg_color + "\" \"" + outfile + "\"";

    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("ffmpeg resize and pad failed: " + cmd);
    }

    GENIEX_LOG_DEBUG("resize and pad image completed successfully, output: {}", fmt::to_string(outfile));
    return outfile;
}

/**
 * @brief Concatenate and resample audio files
 */
inline std::string concat_and_resample_audio(
    const char** inputs, int32_t audio_count, int sampleRate = 16000, int channels = 1) {
    if (inputs == nullptr || audio_count <= 0) {
        throw std::invalid_argument("No input files provided");
    }

    auto tmpdir   = std::filesystem::temp_directory_path();
    auto listfile = tmpdir / "concat_list.txt";
    auto outfile  = tmpdir / "concat_resampled.wav";

    std::ofstream lf(listfile);
    if (!lf.is_open()) {
        throw std::runtime_error("Failed to create concat list file");
    }

    for (int32_t i = 0; i < audio_count; i++) {
        // Convert to absolute path so ffmpeg can find the file
        auto abs_path = std::filesystem::absolute(inputs[i]);
        lf << "file '" << abs_path.string() << "'\n";
    }
    lf.close();

    std::string cmd =
        "ffmpeg -f concat -safe 0 -hide_banner -loglevel error -y "
        "-i \"" +
        listfile.string() +
        "\" "
        "-ar " +
        std::to_string(sampleRate) + " -ac " + std::to_string(channels) + " \"" + outfile.string() + "\"";

    GENIEX_LOG_DEBUG("concat and resample audios ffmpeg command: {}", fmt::to_string(cmd));
    int ret = std::system(cmd.c_str());
    if (ret != 0) {
        throw std::runtime_error("ffmpeg concat failed, command: " + cmd);
    }

    return outfile.string();
}

}  // namespace image_utils
}  // namespace geniex
