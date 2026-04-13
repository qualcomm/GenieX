add_subdirectory(./src)

# Copy outputs based on platform
if(WIN32)
    # Ensure common libs are copied before building this plugin
    add_dependencies(geniex_npu copy_common_libs copy_llama_libs)
endif()

copy_output(out/${GENIEX_PLUGIN_ID_QNN} geniex_npu)

# Copy platform-specific libraries
if(ANDROID)
    add_custom_target(copy_output_qnn_libs ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/libs/android-aarch64 ${CMAKE_BINARY_DIR}/out/${GENIEX_PLUGIN_ID_QNN}/)
elseif(WIN32)
    add_custom_target(copy_output_qnn_libs ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/libs/windows-arm64/dll ${CMAKE_BINARY_DIR}/out/${GENIEX_PLUGIN_ID_QNN}/)
elseif(UNIX) # Linux ARM64
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        add_custom_target(copy_output_qnn_libs ALL
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_LIST_DIR}/libs/linux-aarch64 ${CMAKE_BINARY_DIR}/out/${GENIEX_PLUGIN_ID_QNN}/)
    endif()
endif()
