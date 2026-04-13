# Select source files based on platform

file(GLOB_RECURSE QNN_SRCS *.cpp)
add_library(geniex_npu SHARED ${QNN_SRCS})

target_compile_definitions(geniex_npu PUBLIC ML_SHARED)
target_compile_definitions(geniex_npu PRIVATE BRIDGE_VERSION="${GENIEX_BRIDGE_VERSION}")
set_target_properties(geniex_npu PROPERTIES OUTPUT_NAME "geniex_plugin")

target_include_directories(geniex_npu
    PUBLIC
        ${COMMON_INCLUDE}
    PRIVATE
        ../include/)

# Always link the validation interface target; it is empty unless GENIEX_VALIDATION is ON
# Set platform-specific library directory
if (ANDROID)
    target_link_directories(geniex_npu PRIVATE ../libs/android-aarch64)
    target_link_libraries(geniex_npu
        PRIVATE geniex_bridge geniex_validation
        # llm
        liquid-sdk qwen3-4b-sdk llama3-3b-sdk phi4-sdk phi3-5-sdk granite4-sdk granite-nano-sdk
        # vlm
        omni-neural-sdk qwen3vl-sdk autoneural-sdk
        # asr
        parakeet-sdk wav2vec2-sdk
        # cv
        paddleocr-sdk yolov12-sdk rfdetr-sdk rmbg-v2-sdk table-transformer-sdk depth-anything-v2-sdk
        #  embedding
        embed-gemma-sdk embed-neural-sdk
        # rerank
        jina-rerank-sdk
        # diarization
        # pyannote-sdk
    )
elseif (WIN32)
    target_link_directories(geniex_npu PRIVATE ../libs/windows-arm64/lib)
    target_link_libraries(geniex_npu
        PRIVATE geniex_bridge geniex_validation
        # llm
        qwen3-4b-sdk qwen3-8b-sdk llama3-3b-sdk phi4-sdk phi3-5-sdk granite4-sdk liquid-sdk granite-nano-sdk granite4-sdk smolvla-sdk ministral3-3b-sdk hy-mt-sdk
        # vlm
        omni-neural-sdk qwen3vl-sdk autoneural-sdk
        # asr
        parakeet-sdk wav2vec2-sdk
        # cv
        yolov12-sdk paddleocr-sdk convnext-sdk rfdetr-sdk rmbg-v2-sdk realesrgan-sdk table-transformer-sdk depth-anything-v2-sdk
        # embedding
        embed-gemma-sdk embed-neural-sdk
        # rerank
        jina-rerank-sdk
        # diarization
        pyannote-sdk
    )
elseif (UNIX AND (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"))
    target_link_directories(geniex_npu PRIVATE ../libs/linux-aarch64)
    target_link_libraries(geniex_npu
        PRIVATE geniex_bridge geniex_validation
        # patch for rpath
        -Wl,--no-as-needed nexaproc nexa-mm-process nexa-sampling -Wl,--as-needed
        # llm - only models with available libraries
        liquid-sdk granite-nano-sdk
        # vlm
        autoneural-sdk
        # asr - parakeet available
        parakeet-sdk
        # cv
        convnext-sdk yolov12-sdk
        # embedding
        embed-gemma-sdk embed-neural-sdk
        # rerank
        jina-rerank-sdk
        # diarization - not available
        # pyannote-sdk
    )
else()
    message(FATAL_ERROR "QNN plugin: unsupported platform")
endif()
