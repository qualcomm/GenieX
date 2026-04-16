#pragma once

// Standard library includes
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "nexaproc-export.h"
#include "xtensor-all.hpp"

/**
 * @brief High-performance C++ multimodal processing library for AI applications
 *
 * This namespace contains interfaces and utilities for processing audio, vision,
 * and text data for various machine learning models.
 */
namespace mm_process {

/**
 * @brief Multimodal content container
 *
 * Represents different types of multimodal content with associated metadata
 * for processing by various AI models.
 */
struct GENIEXPROC_API MMContent {
    /**
     * @brief Type of multimodal content
     */
    enum class Type {
        TEXT,
        IMAGE,
        VIDEO,
        AUDIO,
    };

    Type type;

    std::string text  = "";
    std::string image = "";
    std::string video = "";
    std::string audio = "";

    // Vision parameters
    int   nframes        = -1;
    float fps            = -1;
    int   min_frames     = -1;
    int   max_frames     = -1;
    int   min_pixels     = -1;
    int   max_pixels     = -1;
    int   total_pixels   = -1;
    int   resized_height = -1;
    int   resized_width  = -1;

    // Audio parameters
    int sample_rate = -1;
    int nchannels   = -1;

    // Video parameters
    double video_start = -1;
    double video_end   = -1;
};

/**
 * @brief Convert MMContentType to string representation
 * @param type The content type to convert
 * @return String representation of the content type
 */
GENIEXPROC_API std::string mm_content_type_to_string(MMContent::Type type);

/**
 * @brief Chat message for conversation-based AI models
 *
 * Supports role-based messaging with mixed multimodal content.
 */
struct GENIEXPROC_API ChatMessage {
    std::string            role;
    std::string            content;
    std::vector<MMContent> mm_contents;

    inline static const std::string ROLE_USER        = "user";
    inline static const std::string ROLE_ASSISTANT   = "assistant";
    inline static const std::string ROLE_SYSTEM      = "system";
    inline static const std::string ROLE_OBSERVATION = "observation";

    ChatMessage() = default;
    ChatMessage(std::string role, std::string content);

    friend GENIEXPROC_API std::ostream &operator<<(std::ostream &os, const ChatMessage &self);
};

/**
 * @brief Unified output format for batch processing
 *
 * Contains tokenized text, image tensors, video tensors, and audio features
 * in a format suitable for multimodal AI model input.
 */
struct GENIEXPROC_API BatchFeatures {
    std::string         text;
    xt::xarray<int32_t> input_ids;
    xt::xarray<int32_t> attention_mask;
    xt::xarray<int32_t> token_type_ids;
    xt::xarray<float>   pixel_values;          ///< Image pixel values (must be uniform tensor due to C++ static typing)
    xt::xarray<int64_t> pixel_attention_mask;  ///< Pixel attention mask (1 for valid pixels, 0 for padding)
    xt::xarray<size_t>  image_grid_thw;
    xt::xarray<float>   pixel_values_videos;
    xt::xarray<size_t>  video_grid_thw;
    xt::xarray<float>   audio_features;
    xt::xarray<int32_t> audio_attention_mask;

    bool verbose = false;

    friend GENIEXPROC_API std::ostream &operator<<(std::ostream &os, const BatchFeatures &self);
};

// Image utilities

/**
 * Load image file and resize to the desired size.
 *
 * @param image_path Path to the image file.
 * @param resized_height Desired height. If -1, use the original height.
 * @param resized_width Desired width. If -1, use the original width.
 * @param channels Desired number of channels. Default is 3.
 * @param interpolation Interpolation method. Default is 4 (STBIR_FILTER_CATMULLROM).
 * @return Image data with shape (H, W, C) where H is the height, W is the width, and C is the number of channels.
 */
GENIEXPROC_API xt::xtensor<uint8_t, 3> load_image(const std::string &image_path, const int64_t resized_height = -1,
    const int64_t resized_width = -1, const int channels = 3, const int interpolation = 4);

// Audio utilities

/**
 * Load audio file and resample to the desired sampling rate.
 *
 * @param audio_path Path to the audio file.
 * @param sampling_rate Desired sampling rate. If -1, use the sampling rate of original audio.
 * @param mono Whether to convert the audio to mono.
 * @param backend Backend to use for audio loading. Default is "sndfile".
 * @return Audio data with shape (T, C) where T is the number of samples and C is the number of channels.
 */
GENIEXPROC_API xt::xtensor<float, 2> load_audio(const std::string &audio_path, const int64_t sampling_rate = -1,
    const bool mono = true, const std::string &backend = "sndfile");

//==============================================================================
// CONSTANTS
//==============================================================================

/** @brief Default audio sampling rate (16kHz) */
const int AUDIO_DEFAULT_SAMPLING_RATE = 16000;

//==============================================================================
// BASE CLASSES
//==============================================================================

/**
 * @brief Base class for sequence feature extractors
 *
 * Provides common functionality for audio processing feature extractors.
 */
class GENIEXPROC_API SequenceFeatureExtractor {
   public:
    virtual ~SequenceFeatureExtractor() = default;

   protected:
    SequenceFeatureExtractor(int feature_size, int sampling_rate = AUDIO_DEFAULT_SAMPLING_RATE,
        float padding_value = 0.0f, const std::string &padding_side = "right");

    // Protected interface for derived classes
    std::tuple<std::vector<xt::xarray<float>>, std::vector<xt::xarray<int32_t>>> pad(
        const std::vector<xt::xtensor<float, 2>> &features, const std::string &padding, size_t max_length,
        const bool truncation = true, const int pad_to_multiple_of = 128);

    int         feature_size;
    int         sampling_rate;
    float       padding_value;
    std::string padding_side;

   private:
    xt::xarray<float> _truncate(
        const xt::xarray<float> &features, size_t max_length, const int pad_to_multiple_of, const bool truncation);
    std::tuple<xt::xarray<float>, xt::xarray<int32_t>> _pad(const xt::xarray<float> &features,
        const std::string &padding_strategy, size_t max_length, const int pad_to_multiple_of);
};

//==============================================================================
// PROCESSING INTERFACES
//==============================================================================

namespace whisper {

/**
 * @brief Abstract interface for Whisper-based audio feature extraction
 */
class GENIEXPROC_API WhisperFeatureExtractor : public SequenceFeatureExtractor {
   public:
    virtual ~WhisperFeatureExtractor() = default;

    /**
     * @brief Extract features from raw audio speech
     * @param raw_speech Vector of audio tensors
     * @param padding Padding strategy ("max_length", "longest", etc.)
     * @param max_length Maximum sequence length (-1 for no limit)
     * @param truncation Whether to truncate sequences
     * @param pad_to_multiple_of Pad length to multiple of this value
     * @return Tuple of (features, attention_mask)
     */
    virtual std::tuple<xt::xarray<float>, xt::xarray<int32_t>> extract_features(
        const std::vector<xt::xtensor<float, 2>> &raw_speech, const std::string &padding = "longest",
        const int max_length = -1, const bool truncation = true, const int pad_to_multiple_of = 128) = 0;

   protected:
    WhisperFeatureExtractor(int feature_size, int sampling_rate, float padding_value)
        : SequenceFeatureExtractor(feature_size, sampling_rate, padding_value) {}
};

/**
 * @brief Factory function for creating WhisperFeatureExtractor instances
 */
GENIEXPROC_API std::unique_ptr<WhisperFeatureExtractor> create_whisper_feature_extractor(int feature_size = 80,
    int sampling_rate = AUDIO_DEFAULT_SAMPLING_RATE, int hop_length = 160, int chunk_length = 30, int n_fft = 400,
    float padding_value = 0.0f, float dither = 0.0f);

}  // namespace whisper

namespace wav2vec2 {

/**
 * @brief Abstract interface for Wav2Vec2 audio feature extraction
 *
 */
class GENIEXPROC_API Wav2Vec2FeatureExtractor {
   public:
    virtual ~Wav2Vec2FeatureExtractor() = default;

    /**
     * @brief Extract features from an audio tensor
     *
     * @param audio_data Audio tensor with shape (T, C) where T=samples, C=channels
     * @return Normalized audio features with shape (1, T)
     */
    virtual xt::xarray<float> extract_features(const xt::xtensor<float, 2> &audio_data) = 0;

    /**
     * @brief Extract features from an audio file
     *
     * @param audio_path Path to audio file
     * @return Normalized audio features with shape (1, T)
     */
    virtual xt::xarray<float> extract_features(const std::string &audio_path) = 0;

   protected:
    Wav2Vec2FeatureExtractor(int sampling_rate, bool do_normalize)
        : sampling_rate(sampling_rate), do_normalize(do_normalize) {}

    int  sampling_rate;
    bool do_normalize;
};

/**
 * @brief Factory function for creating Wav2Vec2FeatureExtractor instances
 *
 * @param sampling_rate Target sampling rate (typically 16000 Hz)
 * @param do_normalize Whether to normalize the audio to zero mean and unit variance
 * @return Unique pointer to Wav2Vec2FeatureExtractor instance
 */
GENIEXPROC_API std::unique_ptr<Wav2Vec2FeatureExtractor> create_wav2vec2_feature_extractor(
    int sampling_rate = AUDIO_DEFAULT_SAMPLING_RATE, bool do_normalize = true);

}  // namespace wav2vec2

namespace qwen2vl {

// Vision processing constants
const int PATCH_SIZE   = 14;
const int IMAGE_FACTOR = PATCH_SIZE * 2;
const int MIN_PIXELS   = 4 * IMAGE_FACTOR * IMAGE_FACTOR;
const int MAX_PIXELS   = 16384 * IMAGE_FACTOR * IMAGE_FACTOR;

const std::vector<float> OPENAI_CLIP_MEAN = {0.48145466, 0.4578275, 0.40821073};
const std::vector<float> OPENAI_CLIP_STD  = {0.26862954, 0.26130258, 0.27577711};

/**
 * @brief Abstract interface for Qwen2-VL image processing
 */
class GENIEXPROC_API Qwen2VLImageProcessor {
   public:
    virtual ~Qwen2VLImageProcessor() = default;

    /**
     * @brief Process images and videos for model input
     * @param images Vector of image tensors (N × H × W × C)
     * @param videos Vector of video tensors (N × T × H × W × C)
     * @return Tuple of (processed_features, grid_info)
     */
    virtual std::tuple<xt::xarray<float>, xt::xarray<size_t>> process(
        const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos = {}) = 0;

    /**
     * @brief Get merge size parameter
     * @return Merge size value
     */
    virtual size_t get_merge_size() const = 0;

   protected:
    Qwen2VLImageProcessor() = default;
};

/**
 * @brief Factory function for creating Qwen2VLImageProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Qwen2VLImageProcessor> create_qwen2vl_image_processor(bool do_resize = true,
    int64_t min_pixels = MIN_PIXELS, int64_t max_pixels = MAX_PIXELS, int resample = 0, bool do_rescale = true,
    float rescale_factor = 1.f / 255.f, bool do_normalize = true, std::vector<float> image_mean = OPENAI_CLIP_MEAN,
    std::vector<float> image_std = OPENAI_CLIP_STD, int patch_size = 14, int temporal_patch_size = 2,
    int merge_size = 2);

}  // namespace qwen2vl

namespace qwen2_5_omni {

/**
 * @brief System prompt for Qwen2.5-Omni model
 */
const std::string QWEN_OMNI_SYS_PROMPT =
    "You are Qwen, "
    "a virtual human developed by the Qwen Team, "
    "Alibaba Group, "
    "capable of perceiving auditory and visual inputs, "
    "as well as generating text and speech.";

/**
 * @brief Apply chat template formatting
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true, bool enable_thinking = true);

/**
 * @brief Process multimedia information from chat messages
 */
GENIEXPROC_API std::tuple<std::vector<xt::xtensor<float, 2>>, std::vector<xt::xtensor<uint8_t, 3>>,
    std::vector<xt::xtensor<uint8_t, 4>>>
             process_mm_info(const std::vector<ChatMessage> &messages, bool use_audio_in_video = true);

/**
 * @brief Abstract interface for Qwen2.5-Omni multimodal processing
 */
class GENIEXPROC_API Qwen2_5OmniProcessor {
   public:
    virtual ~Qwen2_5OmniProcessor() = default;

    /**
     * @brief Process multimodal input for model consumption
     * @param text Input text
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @param audios Vector of audio tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos, const std::vector<xt::xtensor<float, 2>> &audios) = 0;

   protected:
    Qwen2_5OmniProcessor() = default;
};

/**
 * @brief Factory function for creating Qwen2_5OmniProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Qwen2_5OmniProcessor> create_qwen2_5_omni_processor(std::string tokenizer_path);

}  // namespace qwen2_5_omni

namespace qwen3vl {

// Vision processing constants
const int PATCH_SIZE   = 16;
const int IMAGE_FACTOR = PATCH_SIZE * 2;
const int MIN_PIXELS   = 4 * IMAGE_FACTOR * IMAGE_FACTOR;
const int MAX_PIXELS   = 16384 * IMAGE_FACTOR * IMAGE_FACTOR;

const std::vector<float> IMAGENET_STANDARD_MEAN = {0.5, 0.5, 0.5};
const std::vector<float> IMAGENET_STANDARD_STD  = {0.5, 0.5, 0.5};

/**
 * @brief Apply chat template for Qwen3-VL model
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true, bool enable_thinking = true);

/**
 * @brief Abstract interface for Qwen2-VL image processing
 */
class GENIEXPROC_API Qwen2VLImageProcessor {
   public:
    virtual ~Qwen2VLImageProcessor() = default;

    /**
     * @brief Process images and videos for model input
     * @param images Vector of image tensors (N × H × W × C)
     * @param videos Vector of video tensors (N × T × H × W × C)
     * @return Tuple of (processed_features, grid_info)
     */
    virtual std::tuple<xt::xarray<float>, xt::xarray<size_t>> process(
        const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

    /**
     * @brief Get merge size parameter
     * @return Merge size value
     */
    virtual size_t get_merge_size() const = 0;

   protected:
    Qwen2VLImageProcessor() = default;
};

/**
 * @brief Factory function for creating Qwen2VLImageProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Qwen2VLImageProcessor> create_qwen2vl_image_processor(bool do_resize = true,
    int64_t min_pixels = MIN_PIXELS, int64_t max_pixels = MAX_PIXELS, int resample = 0, bool do_rescale = true,
    float rescale_factor = 1.f / 255.f, bool do_normalize = true,
    std::vector<float> image_mean = IMAGENET_STANDARD_MEAN, std::vector<float> image_std = IMAGENET_STANDARD_STD,
    int patch_size = PATCH_SIZE, int temporal_patch_size = 2, int merge_size = 2);

/**
 * @brief Process multimedia information from chat messages
 */
GENIEXPROC_API std::tuple<std::vector<xt::xtensor<uint8_t, 3>>, std::vector<xt::xtensor<uint8_t, 4>>> process_mm_info(
    const std::vector<ChatMessage> &messages);

/**
 * @brief Abstract interface for Qwen3-VL multimodal processing
 */
class GENIEXPROC_API Qwen3VLProcessor {
   public:
    virtual ~Qwen3VLProcessor() = default;

    /**
     * @brief Process multimodal input for model consumption
     * @param text Input text
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos) = 0;

   protected:
    Qwen3VLProcessor() = default;
};

/**
 * @brief Factory function for creating Qwen3VLProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Qwen3VLProcessor> create_qwen3vl_processor(std::string tokenizer_path);

namespace embedding {

const int         MAX_PIXELS          = 1800 * IMAGE_FACTOR * IMAGE_FACTOR;
const std::string DEFAULT_INSTRUCTION = "Represent the user's input.";

/**
 * @brief Abstract interface for Qwen3-VL embedding processing
 *
 * Simplified interface that accepts a single query with optional image/video.
 */
class GENIEXPROC_API Qwen3VLEmbeddingProcessor {
   public:
    virtual ~Qwen3VLEmbeddingProcessor() = default;

    /**
     * @brief Process a single query with optional multimodal content for embedding
     *
     * @param query The text query (if empty and no image/video provided, uses "NULL")
     * @param image_path Optional path to an image file (empty string if not provided)
     * @param video_path Optional path to a video file (empty string if not provided)
     * @param video_start_time Optional start time in seconds for video clipping (-1 means from beginning)
     * @param video_end_time Optional end time in seconds for video clipping (-1 means to end)
     * @param instruction Optional system instruction (uses default if empty)
     * @return Processed batch features ready for embedding model
     */
    virtual BatchFeatures process(const std::string &query, const std::string &image_path = "",
        const std::string &video_path = "", double video_start_time = -1, double video_end_time = -1,
        const std::string &instruction = DEFAULT_INSTRUCTION) = 0;

   protected:
    Qwen3VLEmbeddingProcessor() = default;
};

/**
 * @brief Factory function for creating Qwen3VLEmbeddingProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Qwen3VLEmbeddingProcessor> create_qwen3vl_embedding_processor(std::string tokenizer_path);

}  // namespace embedding

}  // namespace qwen3vl

namespace siglip {

// Image processing constants
const std::vector<float> IMAGE_MEAN = {0.5, 0.5, 0.5};
const std::vector<float> IMAGE_STD  = {0.5, 0.5, 0.5};

/**
 * @brief Abstract interface for SigLIP image processing
 */
class GENIEXPROC_API SiglipImageProcessor {
   public:
    virtual ~SiglipImageProcessor() = default;

    /**
     * @brief Process images for SigLIP model
     * @param images Vector of image tensors
     * @return Processed image features
     */
    virtual xt::xarray<float> process(const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

   protected:
    SiglipImageProcessor() = default;
};

/**
 * @brief Factory function for creating SiglipImageProcessor instances
 */
GENIEXPROC_API std::unique_ptr<SiglipImageProcessor> create_siglip_image_processor(bool do_resize = true,
    std::pair<int, int> size = {768, 768}, int resample = 2, bool do_rescale = true, float rescale_factor = 1.f / 255.f,
    bool do_normalize = false, std::vector<float> image_mean = IMAGE_MEAN, std::vector<float> image_std = IMAGE_STD);

}  // namespace siglip

namespace gemma_3n {

/**
 * @brief Abstract interface for Gemma-3N audio feature extraction
 */
class GENIEXPROC_API Gemma3nAudioFeatureExtractor : public SequenceFeatureExtractor {
   public:
    virtual ~Gemma3nAudioFeatureExtractor() = default;

    /**
     * @brief Extract audio features for Gemma-3N model
     * @param raw_speech Vector of audio tensors
     * @param padding Padding strategy
     * @param max_length Maximum sequence length
     * @param truncation Whether to truncate
     * @param pad_to_multiple_of Pad to multiple of this value
     * @return Tuple of (features, attention_mask)
     */
    virtual std::tuple<xt::xarray<float>, xt::xarray<int32_t>> extract_features(
        const std::vector<xt::xtensor<float, 2>> &raw_speech, const std::string &padding = "longest",
        const int max_length = 480000, const bool truncation = true, const int pad_to_multiple_of = 128) = 0;

   protected:
    Gemma3nAudioFeatureExtractor(int feature_size, int sampling_rate, float padding_value)
        : SequenceFeatureExtractor(feature_size, sampling_rate, padding_value) {}
};

/**
 * @brief Factory function for creating Gemma3nAudioFeatureExtractor instances
 */
GENIEXPROC_API std::unique_ptr<Gemma3nAudioFeatureExtractor> create_gemma3n_audio_feature_extractor(
    int feature_size = 128, int sampling_rate = AUDIO_DEFAULT_SAMPLING_RATE, float padding_value = 0.0f,
    float frame_length_ms = 32.0f, float hop_length_ms = 10.0f, float min_frequency = 125.0f,
    float max_frequency = 7600.0f, float preemphasis = 0.97f, bool preemphasis_htk_flavor = true,
    bool fft_overdrive = true, float dither = 0.0f, float input_scale_factor = 1.0f, float mel_floor = 1e-5f);

/**
 * @brief Apply chat template for Gemma-3N model
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true, bool add_bos_token = true);

/**
 * @brief Process multimedia information for Gemma-3N
 */
GENIEXPROC_API std::tuple<std::vector<xt::xtensor<float, 2>>, std::vector<xt::xtensor<uint8_t, 3>>,
    std::vector<xt::xtensor<uint8_t, 4>>>
             process_mm_info(const std::vector<ChatMessage> &messages);

/**
 * @brief Abstract interface for Gemma-3N multimodal processing
 */
class GENIEXPROC_API Gemma3nProcessor {
   public:
    virtual ~Gemma3nProcessor() = default;

    /**
     * @brief Process multimodal input for Gemma-3N model
     * @param text Input text
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @param audios Vector of audio tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos, const std::vector<xt::xtensor<float, 2>> &audios) = 0;

   protected:
    Gemma3nProcessor() = default;
};

/**
 * @brief Factory function for creating Gemma3nProcessor instances
 */
GENIEXPROC_API std::unique_ptr<Gemma3nProcessor> create_gemma3n_processor(
    std::string tokenizer_path, int audio_seq_length = 188, int image_seq_length = 256);

}  // namespace gemma_3n

namespace parakeet {

/**
 * @brief Abstract interface for Parakeet audio feature extraction
 */
class GENIEXPROC_API ParakeetAudioFeatureExtractor {
   public:
    virtual ~ParakeetAudioFeatureExtractor() = default;

    /**
     * @brief Extract audio features for Parakeet model
     * @param raw_speech Vector of audio tensors
     * @param padding Padding strategy
     * @param max_length Maximum sequence length
     * @param truncation Whether to truncate
     * @param pad_to_multiple_of Pad to multiple of this value
     * @return Tuple of (features, attention_mask)
     */
    virtual xt::xarray<float> extract_features(const xt::xtensor<float, 2> &audio_data) = 0;
    virtual xt::xarray<float> extract_features(const std::string &audio_path)           = 0;
};

/**
 * @brief Factory function for creating ParakeetAudioFeatureExtractor instances
 */
GENIEXPROC_API std::unique_ptr<ParakeetAudioFeatureExtractor> create_parakeet_audio_feature_extractor(
    int sample_rate = 16000, const std::string &normalize_type = "per_feature", float window_size = 0.025,
    float window_stride = 0.01, const std::string &window_type = "hann_numpy", int features = 128, int n_fft = 512,
    float dither = 1e-05, int pad_to = 0, float pad_value = 0.0, float preemph = 0.97, float mag_power = 2.0);

}  // namespace parakeet

namespace pyannote {

// =============================================================================
// Part 1 Structures: Audio to Segmentation Input
// =============================================================================

/**
 * @brief Sliding window parameters for audio chunking
 */
struct GENIEXPROC_API SlidingWindowParams {
    int   window_size;      ///< Samples per chunk (e.g., 160000 for 10s at 16kHz)
    int   step_size;        ///< Samples between chunk starts (e.g., 16000 for 1s)
    int   num_chunks;       ///< Number of complete chunks
    bool  has_last_chunk;   ///< Whether there's an incomplete last chunk
    int   last_chunk_size;  ///< Size of last incomplete chunk (if any)
    float duration;         ///< Chunk duration in seconds
    float step;             ///< Step duration in seconds
};

/**
 * @brief Segmentation input ready for model inference
 */
struct GENIEXPROC_API SegmentationInput {
    std::vector<xt::xarray<float>> batches;            ///< Each batch: (batch_size, num_channels, window_size)
    std::vector<int>               batch_sizes;        ///< Actual size of each batch
    int                            total_chunks;       ///< Total number of chunks
    SlidingWindowParams            params;             ///< Window parameters
    float                          waveform_duration;  ///< Total audio duration in seconds
    xt::xtensor<float, 2>          waveform;           ///< Original waveform (channels, samples)
};

/**
 * @brief Batch of embedding inputs (fbank + masks)
 *
 * This structure is used to pass prepared embedding inputs to the embedding model.
 * It contains mel-filterbank features and speaker masks for batch processing.
 */
struct GENIEXPROC_API EmbeddingBatch {
    xt::xtensor<float, 3> fbank_features;   ///< Shape: (batch_size, num_fbank_frames, num_mel_bins)
    xt::xtensor<float, 2> masks;            ///< Shape: (batch_size, num_mask_frames)
    std::vector<int>      chunk_indices;    ///< Track which chunk each item belongs to
    std::vector<int>      speaker_indices;  ///< Track which speaker each item is
    int                   batch_size;       ///< Actual batch size
    int                   num_frames;       ///< Number of fbank frames (e.g., 998)
    int                   num_mask_frames;  ///< Number of mask frames (e.g., 589 from segmentation)
    int                   num_mel_bins;     ///< Number of mel bins (80)
};

// =============================================================================
// Pyannote API
// =============================================================================

/**
 * @brief Configuration for the complete Pyannote diarization pipeline
 */
struct GENIEXPROC_API PyannoteProcessorConfig {
    // Audio processing parameters (Phase 1)
    int   sample_rate             = 16000;
    float segmentation_duration   = 10.0f;  ///< Chunk duration in seconds
    float segmentation_step       = 1.0f;   ///< Step between chunks in seconds
    int   segmentation_batch_size = 32;     ///< Batch size for segmentation

    // Embedding parameters (Phase 2)
    float segmentation_onset   = 0.5f;  ///< Onset threshold for binarization
    int   embedding_batch_size = 32;    ///< Batch size for embedding extraction

    // PLDA model paths (Phase 3)
    std::string plda_transform_path;
    std::string plda_model_path;

    // Clustering parameters (Phase 3)
    std::string clustering_method    = "centroid";  ///< "centroid", "average", "single", "complete"
    float       clustering_threshold = 0.7f;        ///< Distance threshold for AHC
    float       clustering_Fa        = 1.0f;        ///< VBx: scale sufficient statistics
    float       clustering_Fb        = 1.0f;        ///< VBx: speaker regularization
    int         clustering_max_iters = 20;          ///< VBx: max iterations

    // Post-processing parameters (Phase 3)
    float min_duration_off = 0.0f;  ///< Min gap duration (seconds)
};

/**
 * @brief Input for embedding extraction
 */
struct GENIEXPROC_API EmbeddingInput {
    std::vector<EmbeddingBatch> batches;          ///< Batches of (fbank, mask) pairs
    int                         total_items;      ///< Total number of (chunk, speaker) pairs
    std::vector<int>            chunk_indices;    ///< Chunk index for each item
    std::vector<int>            speaker_indices;  ///< Speaker index for each item
};

/**
 * @brief Speech segment with timestamps and speaker label
 */
struct GENIEXPROC_API SpeechSegment {
    float       start_time;     ///< Start time in seconds
    float       end_time;       ///< End time in seconds
    int         speaker_id;     ///< Cluster ID (0-indexed)
    std::string speaker_label;  ///< Human-readable label (e.g., "SPEAKER_00")
};

/**
 * @brief Final Pyannote diarization output
 */
struct GENIEXPROC_API PyannoteOutput {
    std::vector<SpeechSegment> segments;      ///< Speech segments with speaker labels
    int                        num_speakers;  ///< Number of detected speakers
    float                      duration;      ///< Total audio duration (seconds)

    // Optional debug/intermediate results
    xt::xtensor<float, 2> binary_diarization;  ///< Binary frame-level output (frames × speakers)
    std::vector<int>      cluster_labels;      ///< Cluster assignment for each embedding
};

/**
 * @brief Pyannote processor - handles all 3 phases of speaker diarization
 *
 * This class provides a complete speaker diarization pipeline:
 * - Phase 1: Audio → Segmentation model input
 * - Phase 2: Segmentation output → Embedding model input
 * - Phase 3: Embeddings → Final speaker segments
 *
 * Usage pattern:
 * @code
 * // 1. Create and configure processor
 * auto processor = create_pyannote_processor();
 * PyannoteProcessorConfig config;
 * config.plda_transform_path = "xvec_transform.npz";
 * config.plda_model_path = "plda.npz";
 * config.clustering_threshold = 0.7f;
 *
 * // 2. Prepare segmentation input
 * auto seg_input = processor->prepare_segmentation_input(audio_path, config);
 *
 * // 3. Run segmentation model (user's responsibility)
 * auto seg_logits = run_segmentation_model(seg_input.batches);
 *
 * // 4. Prepare embedding input
 * auto emb_input = processor->prepare_embedding_input(
 *     seg_logits, seg_input.waveform, seg_input.params, config);
 *
 * // 5. Run embedding model (user's responsibility)
 * auto embeddings = run_embedding_model(emb_input.batches);
 *
 * // 6. Process to final output
 * auto result = processor->process_to_output(
 *     embeddings, seg_logits, seg_input.params, config);
 * @endcode
 */
class GENIEXPROC_API PyannoteProcessor {
   public:
    virtual ~PyannoteProcessor() = default;

    // =====================================================================
    // Phase 1: Audio → Segmentation Input
    // =====================================================================

    /**
     * @brief Prepare audio for segmentation model
     *
     * Encapsulates:
     * - Audio loading and resampling
     * - Sliding window chunking
     * - Batching for model inference
     *
     * @param audio_path Path to audio file
     * @param config Processor configuration
     * @return SegmentationInput ready for model inference
     */
    virtual SegmentationInput prepare_segmentation_input(
        const std::string &audio_path, const PyannoteProcessorConfig &config) = 0;

    virtual SegmentationInput prepare_segmentation_input(
        const xt::xtensor<float, 2> &audio_data, const PyannoteProcessorConfig &config) = 0;

    // =====================================================================
    // Phase 2: Segmentation Output → Embedding Input
    // =====================================================================

    /**
     * @brief Prepare segmentation output for embedding model
     *
     * Encapsulates:
     * - Powerset to multilabel conversion
     * - Binary mask computation
     * - Clean mask extraction (overlap removal)
     * - Speaker mask selection
     * - Waveform cropping per chunk
     * - Fbank feature extraction
     * - Batch preparation with masks
     *
     * @param segmentation_logits Segmentation model output (chunks × frames × classes)
     * @param waveform Original waveform (channels × samples)
     * @param params Sliding window parameters from Phase 1
     * @param config Processor configuration
     * @return EmbeddingInput ready for embedding model
     */
    virtual EmbeddingInput prepare_embedding_input(const xt::xtensor<float, 3> &segmentation_logits,
        const xt::xtensor<float, 2> &waveform, const SlidingWindowParams &params,
        const PyannoteProcessorConfig &config) = 0;

    // =====================================================================
    // Phase 3: Embeddings → Final Output
    // =====================================================================

    /**
     * @brief Process embeddings to final diarization output
     *
     * Encapsulates the complete post-processing pipeline:
     * 1. Filter active embeddings (remove silent speakers)
     * 2. PLDA transformation (load model, transform, normalize)
     * 3. Hierarchical clustering (AHC with configurable method)
     * 4. VBx refinement (Variational Bayes clustering)
     * 5. Segmentation reconstruction (map clusters back to chunks)
     * 6. Frame aggregation (overlap-add across chunks)
     * 7. Binarization (apply onset threshold)
     * 8. Segment extraction (continuous speech regions)
     * 9. Speaker label assignment (SPEAKER_00, SPEAKER_01, ...)
     *
     * @param embeddings Embedding model output (chunks × speakers × embedding_dim)
     * @param segmentation_logits Segmentation model output (chunks × frames × classes)
     * @param params Sliding window parameters from Phase 1
     * @param config Processor configuration (must include PLDA paths)
     * @return PyannoteOutput with final speaker segments
     */
    virtual PyannoteOutput process_to_output(const xt::xtensor<float, 3> &embeddings,
        const xt::xtensor<float, 3> &segmentation_logits, const SlidingWindowParams &params,
        const PyannoteProcessorConfig &config) = 0;

   protected:
    PyannoteProcessor() = default;
};

/**
 * @brief Factory function for PyannoteProcessor
 *
 * Creates a new processor instance. The processor is stateless and can be
 * reused for multiple audio files.
 *
 * @return Unique pointer to PyannoteProcessor instance
 */
GENIEXPROC_API std::unique_ptr<PyannoteProcessor> create_pyannote_processor();

}  // namespace pyannote

namespace smolvlm {

// Image processing constants
const std::vector<float> IMAGE_MEAN = {0.5f, 0.5f, 0.5f};
const std::vector<float> IMAGE_STD  = {0.5f, 0.5f, 0.5f};

/**
 * @brief Result structure for SmolVLM image processing
 */
struct GENIEXPROC_API SmolVLMImageProcessorOutput {
    xt::xarray<float> pixel_values;  ///< Processed images (num_images, C, H, W)
    std::vector<int>  rows;          ///< Number of row splits per image (0 if no splitting)
    std::vector<int>  cols;          ///< Number of column splits per image (0 if no splitting)
};

/**
 * @brief Abstract interface for SmolVLM image processing
 *
 * SmolVLM uses SigLIP-style image processing with optional image splitting
 * for handling high-resolution images.
 */
class GENIEXPROC_API SmolVLMImageProcessor {
   public:
    virtual ~SmolVLMImageProcessor() = default;

    /**
     * @brief Process images for SmolVLM model
     * @param images Vector of image tensors (HWC format)
     * @param return_row_col_info Whether to return row/col split information
     * @return SmolVLMImageProcessorOutput with processed images and split info
     */
    virtual SmolVLMImageProcessorOutput process(
        const std::vector<xt::xtensor<uint8_t, 3>> &images, bool return_row_col_info = true) = 0;

    /**
     * @brief Get the image sequence length (number of image tokens per image)
     * @return Image sequence length
     */
    virtual int get_image_seq_len() const = 0;

   protected:
    SmolVLMImageProcessor() = default;
};

/**
 * @brief Factory function for creating SmolVLMImageProcessor instances
 *
 * Default parameters match HuggingFace Transformers SmolVLMImageProcessor:
 * - size: 512 (vision encoder input size)
 * - do_image_splitting: true (automatically split large images into patches)
 *
 * @param size Target image size (default: 512, matches Transformers)
 * @param do_image_splitting Whether to enable image splitting for high-res images (default: true, matches Transformers)
 * @param patch_size Vision encoder patch size (default: 14)
 * @param scale_factor Scale factor for reducing sequence length (default: 2)
 * @param resample Interpolation method (default: 4 for CATMULLROM, closest to PIL LANCZOS)
 * @param do_rescale Whether to rescale to [0, 1]
 * @param rescale_factor Rescale factor (default: 1/255)
 * @param do_normalize Whether to normalize with mean/std
 * @param image_mean Mean values for normalization
 * @param image_std Std values for normalization
 * @return Unique pointer to SmolVLMImageProcessor
 */
GENIEXPROC_API std::unique_ptr<SmolVLMImageProcessor> create_smolvlm_image_processor(int size = 512,
    bool do_image_splitting = true, int patch_size = 14, int scale_factor = 2, int resample = 4, bool do_rescale = true,
    float rescale_factor = 1.f / 255.f, bool do_normalize = true, std::vector<float> image_mean = IMAGE_MEAN,
    std::vector<float> image_std = IMAGE_STD);

/**
 * @brief Apply chat template for SmolVLM model
 *
 * @param messages Vector of chat messages
 * @param add_generation_prompt Whether to add generation prompt
 * @return Formatted chat string
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true);

/**
 * @brief Process multimedia information from chat messages
 *
 * @param messages Vector of chat messages
 * @return Vector of loaded images
 */
GENIEXPROC_API std::vector<xt::xtensor<uint8_t, 3>> process_image_info(const std::vector<ChatMessage> &messages);

/**
 * @brief Abstract interface for SmolVLM multimodal processing
 */
class GENIEXPROC_API SmolVLMProcessor {
   public:
    virtual ~SmolVLMProcessor() = default;

    /**
     * @brief Process multimodal input for SmolVLM model
     * @param text Input text (may contain <image> or <video> tokens)
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos) = 0;

    /**
     * @brief Expand image tokens in text based on row/col splits
     *
     * For each <image> token, expands to appropriate sequence based on splitting.
     *
     * @param text Input text with <image> tokens
     * @param image_rows Vector of row splits per image (0 if not split)
     * @param image_cols Vector of column splits per image (0 if not split)
     * @return Expanded text with image token sequences
     */
    virtual std::string expand_text_with_image_tokens(
        const std::string &text, const std::vector<int> &image_rows, const std::vector<int> &image_cols) = 0;

   protected:
    SmolVLMProcessor() = default;
};

/**
 * @brief Factory function for creating SmolVLMProcessor instances
 *
 * Default parameters match HuggingFace Transformers SmolVLMProcessor:
 * - image_size: 512 (vision encoder input size)
 * - do_image_splitting: true (automatically split large images)
 *
 * @param tokenizer_path Path to tokenizer JSON file
 * @param image_size Target image size (default: 512, matches Transformers)
 * @param do_image_splitting Whether to enable image splitting for high-res images (default: true, matches Transformers)
 * @return Unique pointer to SmolVLMProcessor
 */
GENIEXPROC_API std::unique_ptr<SmolVLMProcessor> create_smolvlm_processor(
    const std::string &tokenizer_path, int image_size = 512, bool do_image_splitting = true);

}  // namespace smolvlm

namespace deepseek_ocr {

//==============================================================================
// STRUCTURES
//==============================================================================

/**
 * @brief Configuration for DeepSeek OCR preprocessing
 */
struct GENIEXPROC_API DeepSeekOCRConfig {
    int  base_size        = 1024;  ///< Base size for global image view
    int  image_size       = 640;   ///< Size for cropped image patches
    bool crop_mode        = true;  ///< Enable dynamic cropping
    int  patch_size       = 16;    ///< Vision transformer patch size
    int  downsample_ratio = 4;     ///< Downsampling ratio for tokens
    int  min_num          = 2;     ///< Minimum crop grid size
    int  max_num          = 9;     ///< Maximum crop grid size
};

/**
 * @brief Preprocessed output ready for DeepSeek OCR model inference
 */
struct GENIEXPROC_API DeepSeekOCRPreprocessed {
    // Tokenization outputs
    xt::xarray<int32_t> input_ids;  ///< Shape: (sequence_length,)

    // Image tensors
    xt::xarray<float>   images_ori;           ///< Shape: (num_images, 3, image_size, image_size)
    xt::xarray<float>   images_crop;          ///< Shape: (num_crops, 3, base_size, base_size)
    xt::xarray<int32_t> images_spatial_crop;  ///< Shape: (num_images, 2) - [width_crop_num, height_crop_num]
};

//==============================================================================
// FUNCTIONS
//==============================================================================

/**
 * @brief Apply chat template formatting for DeepSeek OCR
 *
 * Formats chat messages into a prompt string compatible with DeepSeek OCR model.
 * Implements the 'plain' format from the Python code's format_messages function.
 *
 * @param messages Vector of chat messages
 * @param system_prompt Optional system prompt (default: empty)
 * @return Formatted prompt string
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, const std::string &system_prompt = "");

//==============================================================================
// PROCESSOR INTERFACE
//==============================================================================

/**
 * @brief DeepSeek OCR preprocessor interface
 *
 * Handles image preprocessing and tokenization for DeepSeek OCR model.
 * Supports dynamic image cropping for high-resolution documents.
 *
 * Usage example:
 * @code
 * auto processor = create_deepseek_ocr_processor("tokenizer.json");
 *
 * DeepSeekOCRConfig config;
 * config.base_size = 1024;
 * config.image_size = 640;
 * config.crop_mode = true;
 *
 * std::string prompt = "<image>\nExtract all text from the image.";
 * auto image = load_image("document.png");
 *
 * auto result = processor->preprocess(prompt, {image}, config);
 * // Use result.input_ids, result.images_ori, etc. for model inference
 * @endcode
 */
class GENIEXPROC_API DeepSeekOCRProcessor {
   public:
    virtual ~DeepSeekOCRProcessor() = default;

    /**
     * @brief Preprocess prompt and images for DeepSeek OCR model
     *
     * This method handles:
     * - Text tokenization with image token insertion
     * - Image loading and transformation
     * - Dynamic cropping (if enabled)
     * - Image resizing and padding
     * - Tensor preparation for model input
     *
     * @param prompt Text prompt (can include <image> placeholders)
     * @param images Vector of input images (H × W × C, RGB format)
     * @param config Preprocessing configuration
     * @return DeepSeekOCRPreprocessed structure with all model inputs
     */
    virtual DeepSeekOCRPreprocessed preprocess(const std::string &prompt,
        const std::vector<xt::xtensor<uint8_t, 3>> &images, const DeepSeekOCRConfig &config = DeepSeekOCRConfig()) = 0;

    /**
     * @brief Preprocess with image paths (convenience method)
     *
     * @param prompt Text prompt
     * @param image_paths Vector of image file paths
     * @param config Preprocessing configuration
     * @return DeepSeekOCRPreprocessed structure
     */
    virtual DeepSeekOCRPreprocessed preprocess(const std::string &prompt, const std::vector<std::string> &image_paths,
        const DeepSeekOCRConfig &config = DeepSeekOCRConfig()) = 0;

   protected:
    DeepSeekOCRProcessor() = default;
};

/**
 * @brief Factory function for creating DeepSeekOCRProcessor instances
 *
 * @param tokenizer_path Path to the tokenizer JSON file
 * @return Unique pointer to DeepSeekOCRProcessor instance
 */
GENIEXPROC_API std::unique_ptr<DeepSeekOCRProcessor> create_deepseek_ocr_processor(const std::string &tokenizer_path);

}  // namespace deepseek_ocr

namespace nexa_vlm {

// special tokens
const std::string BOS_TOKEN           = "<|startoftext|>";
const std::string START_OF_TURN_TOKEN = "<|im_start|>";
const std::string END_OF_TURN_TOKEN   = "<|im_end|>";

const std::string IMAGE_TOKEN = "<image_soft_token>";
const std::string BOI_TOKEN   = "<start_of_image>";
const std::string EOI_TOKEN   = "<end_of_image>";

/**
 * @brief Process multimedia information for NexaVLM
 */
GENIEXPROC_API std::tuple<std::vector<xt::xtensor<uint8_t, 3>>, std::vector<xt::xtensor<uint8_t, 4>>> process_mm_info(
    const std::vector<ChatMessage> &messages);

/**
 * @brief Abstract interface for NexaVLM multimodal processing
 */
class GENIEXPROC_API NexaVLMProcessor {
   public:
    virtual ~NexaVLMProcessor() = default;

    /**
     * @brief Apply chat template for NexaVLM model
     * @param messages Vector of chat messages
     * @param add_generation_prompt Whether to add generation prompt
     * @param add_bos_token Whether to add BOS token
     * @return Formatted chat string
     */
    virtual std::string apply_chat_template(
        const std::vector<ChatMessage> &messages, bool add_generation_prompt = true, bool add_bos_token = true) = 0;

    /**
     * @brief Process multimodal input for NexaVLM model
     * @param text Input text
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos) = 0;

   protected:
    NexaVLMProcessor() = default;
};

/**
 * @brief Factory function for creating NexaVLMProcessor instances
 */
GENIEXPROC_API std::unique_ptr<NexaVLMProcessor> create_nexa_vlm_processor(std::string tokenizer_path,
    int image_seq_length = 256, std::string bos_token = BOS_TOKEN,
    std::string start_of_turn_token = START_OF_TURN_TOKEN, std::string end_of_turn_token = END_OF_TURN_TOKEN,
    std::string image_token = IMAGE_TOKEN, std::string boi_token = BOI_TOKEN, std::string eoi_token = EOI_TOKEN);

}  // namespace nexa_vlm

namespace idefics3 {

// Special tokens
const std::string START_OF_ROLE = "<|start_of_role|>";
const std::string END_OF_ROLE   = "<|end_of_role|>";
const std::string END_OF_TEXT   = "<|end_of_text|>";
const std::string IMAGE_TOKEN   = "<image>";

/**
 * @brief Apply chat template for Idefics3 model
 *
 * Formats chat messages according to Idefics3's chat template.
 * Template format: <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n
 *
 * @param messages Vector of chat messages
 * @param add_generation_prompt Whether to add generation prompt for assistant
 * @return Formatted chat string
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true);

/**
 * @brief Result structure for Idefics3 image processing
 */
struct GENIEXPROC_API Idefics3ImageProcessorOutput {
    xt::xarray<float>   pixel_values;          ///< Processed images (batch, num_images, C, H, W)
    xt::xarray<int64_t> pixel_attention_mask;  ///< Pixel attention mask (batch, num_images, H, W)
    std::vector<int>    rows;                  ///< Number of row splits per image (0 if no splitting)
    std::vector<int>    cols;                  ///< Number of column splits per image (0 if no splitting)
};

/**
 * @brief Abstract interface for Idefics3 image processing
 *
 * Processes images with optional image splitting for high-resolution images.
 *
 * @note Default parameters are aligned with the pretrained IBM Granite-Docling model
 *       (ibm-granite/granite-docling-258M)
 */
class GENIEXPROC_API Idefics3ImageProcessor {
   public:
    virtual ~Idefics3ImageProcessor() = default;

    /**
     * @brief Process images for Idefics3 model
     * @param images Vector of image tensors (HWC format)
     * @return Idefics3ImageProcessorOutput with processed images and split info
     */
    virtual Idefics3ImageProcessorOutput process(const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

   protected:
    Idefics3ImageProcessor() = default;
};

/**
 * @brief Factory function for creating Idefics3ImageProcessor instances
 *
 * Default parameters match the pretrained IBM Granite-Docling model.
 *
 * @param do_resize Whether to resize the image (default: true)
 * @param size_longest_edge Max size for initial resize (default: 2048)
 * @param resample Interpolation method (default: 4 for CATMULLROM, closest to PIL LANCZOS)
 * @param do_image_splitting Whether to enable image splitting (default: true)
 * @param max_image_size_longest_edge Size for image patches (default: 512)
 * @param do_rescale Whether to rescale to [0, 1] (default: true)
 * @param rescale_factor Rescale factor (default: 1/255)
 * @param do_normalize Whether to normalize with mean/std (default: true)
 * @param image_mean Mean values for normalization (default: [0.5, 0.5, 0.5])
 * @param image_std Std values for normalization (default: [0.5, 0.5, 0.5])
 * @param do_pad Whether to pad images (default: true, must be true for C++ implementation)
 * @return Unique pointer to Idefics3ImageProcessor
 */
GENIEXPROC_API std::unique_ptr<Idefics3ImageProcessor> create_idefics3_image_processor(bool do_resize = true,
    int size_longest_edge = 2048, int resample = 4, bool do_image_splitting = true,
    int max_image_size_longest_edge = 512, bool do_rescale = true, float rescale_factor = 1.f / 255.f,
    bool do_normalize = true, std::vector<float> image_mean = {0.5f, 0.5f, 0.5f},
    std::vector<float> image_std = {0.5f, 0.5f, 0.5f}, bool do_pad = true);

/**
 * @brief Abstract interface for Idefics3 multimodal processing
 */
class GENIEXPROC_API Idefics3Processor {
   public:
    virtual ~Idefics3Processor() = default;

    /**
     * @brief Process multimodal input for Idefics3 model
     * @param text Input text (may contain <image> tokens)
     * @param images Vector of image tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

   protected:
    Idefics3Processor() = default;
};

/**
 * @brief Factory function for creating Idefics3Processor instances
 *
 * @param tokenizer_path Path to tokenizer JSON file
 * @param image_seq_len Number of image tokens per image patch (default: 64)
 * @param do_image_splitting Whether to enable image splitting for high-res images (default: true)
 * @param image_processor Optional custom image processor. If nullptr, a default processor will be created
 * @return Unique pointer to Idefics3Processor
 *
 * @note Default parameters are aligned with the pretrained IBM Granite-Docling model
 */
GENIEXPROC_API std::unique_ptr<Idefics3Processor> create_idefics3_processor(const std::string &tokenizer_path,
    int image_seq_len = 64, bool do_image_splitting = true,
    std::unique_ptr<Idefics3ImageProcessor> image_processor = nullptr);

}  // namespace idefics3

namespace jina_clip {

// Image processing constants
const std::vector<float> OPENAI_CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
const std::vector<float> OPENAI_CLIP_STD  = {0.26862954f, 0.26130258f, 0.27577711f};

/**
 * @brief Abstract interface for Jina-CLIP multimodal processing
 */
class GENIEXPROC_API JinaClipProcessor {
   public:
    virtual ~JinaClipProcessor() = default;

    /**
     * @brief Process text input only
     * @param text Input text string
     * @param model Model version: "v1" or "v2" (default: "v1")
     * @return Processed batch features with input_ids and attention_mask
     *
     * v1 (BertTokenizer): CLS=101, SEP=102, truncates to 256 tokens (no padding)
     * v2 (XLMRobertaTokenizer): BOS=0, EOS=2, truncates to 512 tokens (no padding)
     */
    virtual BatchFeatures process_text(const std::string &text, const std::string &model = "v1") = 0;

    /**
     * @brief Process vision input only
     * @param images Vector of image tensors
     * @return Processed batch features with pixel_values
     */
    virtual BatchFeatures process_vision(const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

    /**
     * @brief Process multimodal input for Jina-CLIP models
     * @param text Input text string
     * @param images Vector of image tensors
     * @param model Model version: "v1" or "v2" (default: "v1")
     * @return Processed batch features
     *
     * v1 (BertTokenizer): CLS=101, SEP=102, truncates to 256 tokens (no padding)
     * v2 (XLMRobertaTokenizer): BOS=0, EOS=2, truncates to 512 tokens (no padding)
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::string &model = "v1") = 0;

   protected:
    JinaClipProcessor() = default;
};

/**
 * @brief Factory function for creating JinaClipProcessor instances
 *
 * Creates a processor that supports both v1 and v2 models.
 * Pass model="v1" or model="v2" to the process() method to specify the version.
 *
 * Default parameters match the Jina-CLIP v1 runtime configuration.
 *
 * @param tokenizer_path Path to tokenizer JSON file (v1-tokenizer.json or v2-tokenizer.json)
 * @param image_size Target image size (default: 224 for v1, 512 for v2)
 * @param image_mean Mean values for normalization (default: OPENAI_CLIP_MEAN)
 * @param image_std Std values for normalization (default: OPENAI_CLIP_STD)
 * @param resize_mode Resize strategy: "shortest", "longest", "squash" (default: "shortest")
 * @param interpolation Interpolation filter: 2=bilinear, 3=bicubic, 4=bicubic (default: 4)
 * @param fill_color Fill color for padding in "longest" mode (default: 0)
 * @return Unique pointer to JinaClipProcessor
 */
GENIEXPROC_API std::unique_ptr<JinaClipProcessor> create_jina_clip_processor(const std::string &tokenizer_path,
    int                image_size = 224,  // v1 default
    std::vector<float> image_mean = OPENAI_CLIP_MEAN, std::vector<float> image_std = OPENAI_CLIP_STD,
    const std::string &resize_mode   = "shortest",
    int                interpolation = 4,  // 4 = STBIR_FILTER_CATMULLROM   (bicubic)
    int                fill_color    = 0);

}  // namespace jina_clip

// ============================================================================
// VIDEOCLIP PROCESSOR
// ============================================================================

namespace videoclip {

// Default normalization parameters for VideoClip
// Reference: Standard ImageNet normalization used by most video models
const std::vector<float> VIDEO_MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> VIDEO_STD  = {0.229f, 0.224f, 0.225f};

/**
 * @brief Abstract interface for VideoClip multimodal processing
 *
 * VideoClip is a video-text model based on CLIP architecture.
 * It processes video frames and text for tasks like video-text matching,
 * video classification, and video retrieval.
 */
class GENIEXPROC_API VideoClipProcessor {
   public:
    virtual ~VideoClipProcessor() = default;

    /**
     * @brief Process text input only
     * @param text Input text string
     * @return Processed batch features with input_ids and attention_mask
     *
     * Tokenizes text with CLIP tokenizer (max length 77 tokens)
     */
    virtual BatchFeatures process_text(const std::string &text) = 0;

    /**
     * @brief Process video file
     * @param video_path Path to video file
     * @param start_time Start time in seconds (optional, processes whole video if not specified)
     * @param end_time End time in seconds (optional, processes whole video if not specified)
     * @param num_frames Number of frames to sample (default: 8)
     * @param target_size Target frame size (default: 224)
     * @return Processed batch features with pixel_values_videos tensor (1, num_frames, 3, H, W)
     *
     * Processes video with:
     * - Extract frames from time range using Decord
     * - Resize to target size
     * - Normalize with ImageNet mean/std [0.485, 0.456, 0.406] / [0.229, 0.224, 0.225]
     * - Convert to (1, num_frames, 3, H, W) format
     */
    virtual BatchFeatures process_video(const std::string &video_path, float start_time = -1.0f, float end_time = -1.0f,
        int num_frames = 8, int target_size = 224) = 0;

    /**
     * @brief Process both text and video
     * @param text Input text string
     * @param video_path Path to video file
     * @param start_time Start time in seconds (optional)
     * @param end_time End time in seconds (optional)
     * @param num_frames Number of frames to sample (default: 8)
     * @param target_size Target frame size (default: 224)
     * @return Processed batch features with input_ids and pixel_values_videos
     */
    virtual BatchFeatures process(const std::string &text, const std::string &video_path, float start_time = -1.0f,
        float end_time = -1.0f, int num_frames = 8, int target_size = 224) = 0;

   protected:
    VideoClipProcessor() = default;
};

/**
 * @brief Factory function to create a VideoClip processor
 *
 * @param tokenizer_path Path to tokenizer.json file
 * @param frame_size Target frame size (default: 224)
 * @param video_mean Mean values for video frame normalization (default: VIDEO_MEAN [0.485, 0.456, 0.406])
 * @param video_std Std values for video frame normalization (default: VIDEO_STD [0.229, 0.224, 0.225])
 * @return Unique pointer to VideoClipProcessor
 */
GENIEXPROC_API std::unique_ptr<VideoClipProcessor> create_videoclip_processor(const std::string &tokenizer_path,
    int frame_size = 224, std::vector<float> video_mean = VIDEO_MEAN, std::vector<float> video_std = VIDEO_STD);

}  // namespace videoclip

namespace lfm2audio {

/**
 * @brief Audio preprocessor configuration for LFM2-Audio
 */
struct GENIEXPROC_API Lfm2AudioPreprocessorConfig {
    int         sample_rate    = 16000;          ///< Sample rate (Hz)
    std::string normalize      = "per_feature";  ///< Normalization type ("per_feature", etc.)
    float       window_size    = 0.025f;         ///< Window size in seconds
    float       window_stride  = 0.01f;          ///< Window stride in seconds
    std::string window         = "hann";         ///< Window type ("hann", "hamming", etc.)
    int         features       = 128;            ///< Number of mel features
    int         n_fft          = 512;            ///< FFT size
    bool        log            = true;           ///< Apply log to mel spectrogram
    int         frame_splicing = 1;         ///< Frame splicing factor (currently unsupported; reserved for future use)
    float       dither         = 1e-5f;     ///< Dithering amount (currently unsupported/reserved for future use)
    float       preemph        = 0.97f;     ///< Preemphasis coefficient
    int         pad_to         = 0;         ///< Pad to length (0 = no padding)
    float       pad_value      = 0.0f;      ///< Padding value
    bool        exact_pad      = false;     ///< Use exact padding for STFT
    float       mag_power      = 2.0f;      ///< Magnitude power (2.0 for power spectrum)
    double      lowfreq        = 0.0;       ///< Lowest frequency in mel filterbank (Hz)
    double      highfreq       = 0.0;       ///< Highest frequency in mel filterbank (Hz, 0 means sr/2)
    std::string mel_norm       = "slaney";  ///< Mel filterbank normalization ("slaney", etc.)
};

/**
 * @brief Model inputs for LFM2-Audio inference
 *
 * Contains all tensors needed for model forward pass.
 */
struct GENIEXPROC_API ChatStateContent {
    xt::xarray<int32_t> text;           ///< Text tokens (1 × text_length)
    xt::xarray<float>   audio_in;       ///< Audio input mel features (128 × time_steps)
    xt::xarray<int64_t> audio_in_lens;  ///< Audio input segment lengths (num_segments,)
    xt::xarray<int32_t> audio_out;      ///< Audio output tokens (codebooks × audio_length)
    xt::xarray<int32_t> modality_flag;  ///< Modality flags (1 × total_length)
};

/**
 * @brief Chat state container for LFM2-Audio conversation management
 *
 * Manages conversation state including text tokens, audio inputs, audio outputs,
 * and modality flags for multi-turn audio-text conversations.
 */
class GENIEXPROC_API Lfm2AudioChatState {
   public:
    virtual ~Lfm2AudioChatState() = default;

    /**
     * @brief Add text to the conversation state
     * @param text Text string to add
     */
    virtual void add_text(const std::string &text) = 0;

    /**
     * @brief Add audio to the conversation state
     * @param audio_path Path to audio file
     * @param sampling_rate Sampling rate to load the audio in (default: 16000)
     */
    virtual void add_audio(const std::string &audio_path, int sampling_rate = 16000) = 0;

    /**
     * @brief End the current conversation turn
     *
     * Adds the end-of-turn token (<|im_end|>)
     */
    virtual void end_turn() = 0;

    /**
     * @brief Start a new conversation turn
     * @param role Role for the turn ("system", "user", or "assistant")
     *
     * Adds the start-of-turn token with role (<|im_start|>{role})
     */
    virtual void new_turn(const std::string &role) = 0;

    /**
     * @brief Append model outputs to the state
     * @param text Generated text tokens (1 × text_length)
     * @param audio_out Generated audio tokens (codebooks × audio_length)
     * @param modality_flag Modality flags (1 × total_length)
     */
    virtual void append(const xt::xtensor<int32_t, 2> &text, const xt::xtensor<int32_t, 2> &audio_out,
        const xt::xtensor<int32_t, 2> &modality_flag) = 0;

    /**
     * @brief Get current chat state for model inference
     *
     * Returns all model inputs needed for inference.
     * Equivalent to unpacking **chat in Python.
     *
     * @return ChatStateContent containing all tensors for model forward pass
     */
    virtual ChatStateContent get_current_chat_state() const = 0;

   protected:
    Lfm2AudioChatState() = default;
};

/**
 * @brief Factory function for creating Lfm2AudioChatState instances
 *
 * Creates a chat state with text tokenizer and audio preprocessor.
 *
 * @param tokenizer_path Path to text tokenizer JSON file
 * @param audio_config Audio preprocessing configuration, default matches the config.json in the
 * LiquidAI/LFM2-Audio-1.5B hf repo
 * @param codebooks Number of audio codebooks (default: 8)
 * @return Unique pointer to Lfm2AudioChatState
 */
GENIEXPROC_API std::unique_ptr<Lfm2AudioChatState> create_lfm2audio_chat_state(const std::string &tokenizer_path,
    const Lfm2AudioPreprocessorConfig &audio_config = Lfm2AudioPreprocessorConfig(), int codebooks = 8);

}  // namespace lfm2audio

namespace ministral3 {

// Default image processing parameters for Ministral3
const int DEFAULT_PATCH_SIZE         = 14;
const int DEFAULT_MAX_IMAGE_SIZE     = 1540;
const int DEFAULT_SPATIAL_MERGE_SIZE = 2;

// CLIP/OpenAI normalization statistics used by Mistral models
const std::vector<float> CLIP_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
const std::vector<float> CLIP_STD  = {0.26862954f, 0.26130258f, 0.27577711f};

/**
 * @brief Apply chat template for Ministral3 model
 */
GENIEXPROC_API std::string apply_chat_template(
    const std::vector<ChatMessage> &messages, bool add_generation_prompt = true);

/**
 * @brief Generate image token string for the given grid dimensions.
 *
 * The token pattern is:
 * - Each row: [IMG] * width_tokens + [IMG_BREAK]
 * - Repeated for height_tokens rows
 * - Last [IMG_BREAK] replaced with [IMG_END]
 *
 * Total tokens = (width_tokens + 1) * height_tokens
 *
 * @param width_tokens Number of tokens in width dimension
 * @param height_tokens Number of tokens in height dimension
 * @return String of concatenated image tokens
 */
GENIEXPROC_API std::string generate_image_tokens(int width_tokens, int height_tokens);

/**
 * @brief Abstract interface for Ministral3 image processing
 *
 * Replicates the exact image preprocessing done by mistral-common library.
 * Uses CLIP/OpenAI normalization and cubic interpolation for resizing.
 *
 * Output BatchFeatures contains:
 * - pixel_values: (B, C, H, W) tensor of processed images
 * - image_grid_thw: (N, 2) array where each row is [width_tokens, height_tokens]
 */
class GENIEXPROC_API Ministral3ImageProcessor {
   public:
    virtual ~Ministral3ImageProcessor() = default;

    /**
     * @brief Process images for model input
     * @param images Vector of image tensors (H × W × C format)
     * @return BatchFeatures with pixel_values and image_grid_thw
     */
    virtual BatchFeatures process(const std::vector<xt::xtensor<uint8_t, 3>> &images) = 0;

    /**
     * @brief Get spatial merge size parameter
     * @return Spatial merge size value
     */
    virtual int get_spatial_merge_size() const = 0;

    /**
     * @brief Get patch size parameter
     * @return Patch size value
     */
    virtual int get_patch_size() const = 0;

   protected:
    Ministral3ImageProcessor() = default;
};

/**
 * @brief Factory function for creating Ministral3ImageProcessor instances
 *
 * Default parameters match the Mistral/Pixtral model configuration.
 *
 * @param image_patch_size Size of each image patch (default: 14)
 * @param max_image_size Maximum dimension for image (default: 1540)
 * @param spatial_merge_size Spatial merging factor (default: 2)
 * @param resample Interpolation method (default: 3 for cubic)
 * @param do_rescale Whether to rescale to [0, 1]
 * @param rescale_factor Rescale factor (default: 1/255)
 * @param do_normalize Whether to normalize with mean/std
 * @param image_mean Mean values for CLIP normalization
 * @param image_std Std values for CLIP normalization
 * @return Unique pointer to Ministral3ImageProcessor
 */
GENIEXPROC_API std::unique_ptr<Ministral3ImageProcessor> create_ministral3_image_processor(
    int image_patch_size = DEFAULT_PATCH_SIZE, int max_image_size = DEFAULT_MAX_IMAGE_SIZE,
    int spatial_merge_size = DEFAULT_SPATIAL_MERGE_SIZE, int resample = 4, bool do_rescale = true,
    float rescale_factor = 1.f / 255.f, bool do_normalize = true, std::vector<float> image_mean = CLIP_MEAN,
    std::vector<float> image_std = CLIP_STD);

/**
 * @brief Process multimedia information from chat messages
 */
GENIEXPROC_API std::tuple<std::vector<xt::xtensor<uint8_t, 3>>, std::vector<xt::xtensor<uint8_t, 4>>> process_mm_info(
    const std::vector<ChatMessage> &messages);

/**
 * @brief Abstract interface for Ministral3 multimodal processing
 */
class GENIEXPROC_API Ministral3Processor {
   public:
    virtual ~Ministral3Processor() = default;

    /**
     * @brief Process multimodal input for model consumption
     * @param text Input text
     * @param images Vector of image tensors
     * @param videos Vector of video tensors
     * @return Processed batch features
     */
    virtual BatchFeatures process(const std::string &text, const std::vector<xt::xtensor<uint8_t, 3>> &images,
        const std::vector<xt::xtensor<uint8_t, 4>> &videos) = 0;

   protected:
    Ministral3Processor() = default;
};

/**
 * @brief Factory function for creating Ministral3Processor instances
 *
 * @param tokenizer_path Path to tokenizer JSON file
 * @param image_patch_size Size of each image patch (default: 14)
 * @param max_image_size Maximum dimension for image (default: 1540)
 * @param spatial_merge_size Spatial merging factor (default: 2)
 * @return Unique pointer to Ministral3Processor
 */
GENIEXPROC_API std::unique_ptr<Ministral3Processor> create_ministral3_processor(std::string tokenizer_path,
    int image_patch_size = DEFAULT_PATCH_SIZE, int max_image_size = DEFAULT_MAX_IMAGE_SIZE,
    int spatial_merge_size = DEFAULT_SPATIAL_MERGE_SIZE);

}  // namespace ministral3

namespace neutts {

/**
 * @brief Abstract interface for NeuTTS-Air text-to-speech processing
 */
class GENIEXPROC_API NeuTtsProcessor {
   public:
    virtual ~NeuTtsProcessor() = default;

    /**
     * @brief Apply chat template for NeuTTS-Air model
     *
     * Creates tokenized prompt following the NeuTTS-Air template.
     * The template structure is:
     * @code
     * user: Convert the text to speech:<|TEXT_PROMPT_START|>{phonemized_text}<|TEXT_PROMPT_END|>
     * assistant:<|SPEECH_GENERATION_START|>{ref_codes}
     * @endcode
     *
     * @param ref_codes Reference audio codes (1D array of speech token IDs)
     * @param ref_text Reference text (transcript of reference audio)
     * @param input_text Input text to synthesize
     * @return Vector of token IDs ready for model input
     */
    virtual std::vector<int32_t> apply_chat_template(
        const xt::xarray<int32_t> &ref_codes, const std::string &ref_text, const std::string &input_text) = 0;

    /**
     * @brief Prepare reference audio for NeuCodec encoding
     *
     * Loads audio at 16kHz mono and pads to multiple of 320 samples.
     *
     * @param audio_path Path to reference audio file
     * @return Prepared audio tensor (1, T) ready for NeuCodec encoding
     */
    virtual xt::xtensor<float, 2> prepare_audio(const std::string &audio_path) = 0;

    /**
     * @brief Get the token ID for speech generation end token
     * @return Token ID for <|SPEECH_GENERATION_END|>
     */
    virtual int32_t get_speech_end_id() = 0;

    /**
     * @brief Decode token IDs into a string
     * @param token_ids Vector of token IDs to decode
     * @return Decoded string
     */
    virtual std::string decode(const std::vector<int32_t> &token_ids) = 0;

    /**
     * @brief Extract speech token IDs from a decoded string
     *
     * Parses strings like "<|speech_123|><|speech_456|>" and extracts [123, 456]
     *
     * @param codes String containing speech tokens
     * @return Vector of extracted speech IDs
     */
    virtual std::vector<int32_t> get_speech_ids(const std::string &codes) = 0;

   protected:
    NeuTtsProcessor() = default;
};

/**
 * @brief Factory function for creating NeuTtsProcessor instances
 *
 * @param tokenizer_path Path to tokenizer JSON file (for phonemization and special tokens)
 * @return Unique pointer to NeuTtsProcessor instance
 */
GENIEXPROC_API std::unique_ptr<NeuTtsProcessor> create_neutts_processor(const std::string &tokenizer_path);

}  // namespace neutts

namespace seamless_m4t {

/**
 * @brief Abstract interface for SeamlessM4T audio feature extraction
 *
 * This feature extractor extracts mel-filter bank features from raw speech,
 * following the Kaldi-style processing used by Wav2Vec2-BERT 2.0 models.
 */
class GENIEXPROC_API SeamlessM4TFeatureExtractor : public SequenceFeatureExtractor {
   public:
    virtual ~SeamlessM4TFeatureExtractor() = default;

    /**
     * @brief Extract features from raw audio speech
     *
     * @param raw_speech Vector of audio tensors, each with shape (T, C) where T=samples, C=channels
     * @param padding Padding strategy ("max_length", "longest", "do_not_pad")
     * @param pad_to_multiple_of Pad length to multiple of this value (default: 2 for stride compatibility)
     * @param max_length Maximum sequence length (0 for no limit)
     * @param truncation Whether to truncate sequences longer than max_length
     * @param do_normalize_per_mel_bins Whether to normalize per mel bin (default: true)
     * @return BatchFeatures with audio_features and audio_attention_mask populated
     */
    virtual BatchFeatures extract_features(const std::vector<xt::xtensor<float, 2>> &raw_speech,
        const std::string &padding = "longest", int pad_to_multiple_of = 2, size_t max_length = 0,
        bool truncation = false, bool do_normalize_per_mel_bins = true) = 0;

   protected:
    SeamlessM4TFeatureExtractor(int feature_size, int sampling_rate, float padding_value)
        : SequenceFeatureExtractor(feature_size, sampling_rate, padding_value) {}
};

/**
 * @brief Factory function for creating SeamlessM4TFeatureExtractor instances
 *
 * @param feature_size Feature dimension (default: 80)
 * @param sampling_rate Expected sampling rate (default: 16000)
 * @param num_mel_bins Number of mel frequency bins (default: 80)
 * @param padding_value Value for padding (default: 1.0)
 * @param stride Stride for frame concatenation (default: 2)
 * @return Unique pointer to SeamlessM4TFeatureExtractor instance
 */
GENIEXPROC_API std::unique_ptr<SeamlessM4TFeatureExtractor> create_seamless_m4t_feature_extractor(int feature_size = 80,
    int sampling_rate = AUDIO_DEFAULT_SAMPLING_RATE, int num_mel_bins = 80, float padding_value = 1.0f, int stride = 2);

}  // namespace seamless_m4t

}  // namespace mm_process