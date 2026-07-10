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
# This plugin is built once per QAIRT/QNN version (QAIRT_QNN_VERSION, forwarded
# to the geniex-qairt subdir above). Each version installs into its own
# lib/qairt-<ver>/ folder so multiple ABI builds coexist in one SDK; the SDK
# registry scans these subdirs and the runtime picks the one matching the QAIRT
# libs passed via --qnn-lib / GENIEX_QNN_LIB. No QNN runtime libs are bundled.
set(QAIRT_PLUGIN_DIR "lib/qairt-${QAIRT_QNN_VERSION}")

set(QAIRT_LIBS geniex_core geniex-proc geniex-proc-vision geniex-proc-audio geniex_vlm)
foreach(dep IN LISTS QAIRT_LIBS)
    if(TARGET ${dep})
        install(TARGETS ${dep}
            LIBRARY DESTINATION ${QAIRT_PLUGIN_DIR}
            RUNTIME DESTINATION ${QAIRT_PLUGIN_DIR})
    endif()
endforeach()
