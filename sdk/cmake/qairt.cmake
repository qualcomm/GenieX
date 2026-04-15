# qairt.cmake - Centralized geniex-qairt build configuration
#
# This module builds third-party/geniex-qairt and provides
# stable targets (geniex_core, etc.) for the qairt plugin to consume.

# Guard against multiple inclusions
if(TARGET geniex_core)
    return()
endif()

set(GENIEX_QAIRT_DIR "${CMAKE_SOURCE_DIR}/../third-party/geniex-qairt"
    CACHE PATH "Path to geniex-qairt source directory")

# Disable examples when embedded; they require QNN SDK headers not present here
set(BUILD_EXAMPLES OFF CACHE BOOL "Build geniex-qairt example executables" FORCE)

# Keep Rust artifact paths short on Windows to avoid MAX_PATH linker failures
# in third-party/tokenizers-cpp when building QAIRT dependencies.
if(WIN32)
    set(TOKENIZERS_CPP_CARGO_TARGET_DIR "${CMAKE_BINARY_DIR}/tok"
        CACHE PATH "Cargo target dir for tokenizers-cpp" FORCE)
endif()

# EXCLUDE_FROM_ALL suppresses third-party install() rules.
# Targets still build because SDK plugins link against them.
add_subdirectory(${GENIEX_QAIRT_DIR} ${CMAKE_BINARY_DIR}/third-party/geniex-qairt EXCLUDE_FROM_ALL)

# Export list of qairt libraries for plugin installation
set(QAIRT_LIBS geniex_core nexaproc nexa-sampling geniex_vlm)
