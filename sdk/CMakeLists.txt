cmake_minimum_required(VERSION 3.16)
# C compiler is required by native dependencies.
project(geniex C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set RPATH configuration for all targets to use relative paths
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Set UTF-8 encoding for MSVC
add_compile_options("$<$<AND:$<C_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")
add_compile_options("$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:/utf-8>")

if(WIN32)
    add_compile_definitions(WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

# Convert CMAKE_TOOLCHAIN_FILE to absolute path if set
if(DEFINED CMAKE_TOOLCHAIN_FILE AND NOT IS_ABSOLUTE "${CMAKE_TOOLCHAIN_FILE}")
    get_filename_component(ABSOLUTE_TOOLCHAIN_PATH "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE)
    set(CMAKE_TOOLCHAIN_FILE ${ABSOLUTE_TOOLCHAIN_PATH})
endif()

set(GENIEX_VERSION "v0.0.0" CACHE STRING "Geniex library version")

# ===============================================================================
# Build Options
# ===============================================================================

option(GENIEX_PLUGIN_LLAMA_CPP  "Enable llama.cpp backend"      ON)
option(GENIEX_PLUGIN_QAIRT      "Enable QAIRT backend"           OFF)
option(GENIEX_DL                "Enable dynamic plugin loading"  ON)
option(GENIEX_DEBUG             "Enable debug mode"              ON)
option(GENIEX_TEST              "Enable unit tests"              ON)

message(STATUS "=============== Options ===============")
message(STATUS "GENIEX_VERSION:          ${GENIEX_VERSION}")
message(STATUS "GENIEX_PLUGIN_LLAMA_CPP: ${GENIEX_PLUGIN_LLAMA_CPP}")
message(STATUS "GENIEX_PLUGIN_QAIRT:     ${GENIEX_PLUGIN_QAIRT}")
message(STATUS "GENIEX_DL:               ${GENIEX_DL}")
message(STATUS "GENIEX_DEBUG:            ${GENIEX_DEBUG}")
message(STATUS "GENIEX_TEST:             ${GENIEX_TEST}")
message(STATUS "=======================================")

include(./cmake/util.cmake)

enable_testing()

set(COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find common libraries (installed with plugin dependencies)
find_common_libs(GENIEX_COMMON_LIB_FILES)

# Plugin IDs
set(GENIEX_PLUGIN_ID_LLAMA_CPP "llama_cpp")
add_compile_definitions(GENIEX_PLUGIN_ID_LLAMA_CPP="${GENIEX_PLUGIN_ID_LLAMA_CPP}")

set(GENIEX_PLUGIN_ID_QAIRT "qairt")
add_compile_definitions(GENIEX_PLUGIN_ID_QAIRT="${GENIEX_PLUGIN_ID_QAIRT}")

include(./cmake/llama_cpp.cmake)

if(GGML_HEXAGON)
    set(_HEXAGON_BDIR ${CMAKE_BINARY_DIR}/third-party/llama.cpp/ggml/src/ggml-hexagon)
    set(_INSTALL_COMMANDS)
    foreach(_V IN ITEMS v68 v69 v73 v75 v79 v81)
        list(APPEND _INSTALL_COMMANDS
            COMMAND ${CMAKE_COMMAND} --install
                ${_HEXAGON_BDIR}/htp-${_V}-prefix/src/htp-${_V}-build
        )
    endforeach()
    add_custom_target(install-htp-skels
        ${_INSTALL_COMMANDS}
        DEPENDS htp-v68 htp-v69 htp-v73 htp-v75 htp-v79 htp-v81
        COMMENT "Installing HTP skel libraries"
    )
    # libggml-htp-cat (inf2cat signing) depends on the .so files being
    # present, so it must run after install-htp-skels.
    if(TARGET libggml-htp-cat)
        add_dependencies(libggml-htp-cat install-htp-skels)
    endif()

    # Install HTP skels and catalog into pkg layout alongside ggml-hexagon.
    set(_HTP_FILES)
    foreach(_V IN ITEMS v68 v69 v73 v75 v79 v81)
        list(APPEND _HTP_FILES ${_HEXAGON_BDIR}/libggml-htp-${_V}.so)
    endforeach()
    install(FILES ${_HTP_FILES} DESTINATION lib/llama_cpp)
    if(WIN32)
        install(FILES ${_HEXAGON_BDIR}/libggml-htp.cat DESTINATION lib/llama_cpp OPTIONAL)
    endif()

    unset(_HEXAGON_BDIR)
    unset(_V)
    unset(_INSTALL_COMMANDS)
    unset(_HTP_FILES)
endif()

if(GENIEX_PLUGIN_QAIRT)
    include(./cmake/qairt.cmake)
endif()

if(GENIEX_DL)
    add_compile_definitions(GENIEX_DL)
endif()

if(GENIEX_DEBUG)
    add_compile_definitions(GENIEX_DEBUG)
    add_compile_definitions(PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(./src)
add_subdirectory(./plugins)

add_subdirectory(./tests)
