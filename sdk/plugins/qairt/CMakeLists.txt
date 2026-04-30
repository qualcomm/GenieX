# setup qairt

set(BUILD_EXAMPLES OFF)
set(GENIEX_BUILD_VLM ON)

# Keep Rust artifact paths short on Windows to avoid MAX_PATH linker failures
# in third-party/tokenizers-cpp when building QAIRT dependencies.
if(WIN32)
    set(TOKENIZERS_CPP_CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/tok")
endif()

add_subdirectory(
    ../../../third-party/geniex-qairt
    ${CMAKE_BINARY_DIR}/third-party/geniex-qairt
    EXCLUDE_FROM_ALL)

# setup plugin

add_subdirectory(./src)

# setup install

set(QAIRT_LIBS geniex_core nexaproc nexa-sampling geniex_vlm)
foreach(dep IN LISTS QAIRT_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/qairt
            RUNTIME DESTINATION lib/qairt)
    endif()
endforeach()

# Install htp-files-v{arch}/ folders from the build tree into lib/qairt/.
install(DIRECTORY "$<TARGET_FILE_DIR:geniex_core>/htp-files/"
    DESTINATION lib/qairt/htp-files
    OPTIONAL)
