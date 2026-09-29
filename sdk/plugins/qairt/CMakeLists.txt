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
#
# One build, one directory: the plugin negotiates the QNN C API at load time, so there
# are no per-ABI variants to install side by side.
set(QAIRT_PLUGIN_DIR "lib/qairt")

set(QAIRT_LIBS geniex_core geniex-proc geniex-proc-vision geniex-proc-audio geniex_vlm)
foreach(dep IN LISTS QAIRT_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION ${QAIRT_PLUGIN_DIR}
            RUNTIME DESTINATION ${QAIRT_PLUGIN_DIR})
    endif()
endforeach()

# Install the bundled HTP runtime from the plugin's build tree. Must land beside
# geniex_core as htp-files/ -- that is where the plugin's resolveHtpPaths() looks.
install(DIRECTORY "$<TARGET_FILE_DIR:geniex_core>/htp-files/"
    DESTINATION ${QAIRT_PLUGIN_DIR}/htp-files
    OPTIONAL)
