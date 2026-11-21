# setup llama.cpp

set(GGML_BLAS OFF)
set(GGML_NATIVE OFF)
set(GGML_BACKEND_DL OFF)

set(LLAMA_BUILD_COMMON ON)
set(LLAMA_CURL OFF)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS ON)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(BUILD_SHARED_LIBS ON)

# lld-link (unlike link.exe) doesn't auto-add oldnames.lib, which upstream
# llama.cpp code needs for POSIX-named CRT calls (isatty/fileno/strdup/...).
# Set as a directory-scoped default so every target under this
# add_subdirectory (llama-common, mtmd, etc.) picks it up.
if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    string(APPEND CMAKE_SHARED_LINKER_FLAGS " -Xlinker /DEFAULTLIB:oldnames.lib")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " -Xlinker /DEFAULTLIB:oldnames.lib")
endif()

add_subdirectory(
    ../../../third-party/llama.cpp
    ${CMAKE_BINARY_DIR}/third-party/llama.cpp
    EXCLUDE_FROM_ALL)

# patch for hexagon signing after disable install
#
# Runs after add_subdirectory so the upstream htp-vNN ExternalProject targets
# and libggml-htp-cat exist and can be referenced here.
if(GGML_HEXAGON)
    set(_HEXAGON_BDIR ${CMAKE_BINARY_DIR}/third-party/llama.cpp/ggml/src/ggml-hexagon)
    get_directory_property(_HTP_PROJECTS DIRECTORY ${_HEXAGON_BDIR} DEFINITION HTP_PROJECTS)
    get_directory_property(_HTP_FILES DIRECTORY ${_HEXAGON_BDIR} DEFINITION HTP_SKELS)
    if(NOT _HTP_PROJECTS)
        message(FATAL_ERROR "hexagon: HTP_PROJECTS not found in ${_HEXAGON_BDIR}")
    endif()
    set(_INSTALL_COMMANDS)
    set(_HTP_PREFIX_DIRS)
    foreach(_P IN LISTS _HTP_PROJECTS)
        list(APPEND _INSTALL_COMMANDS
            COMMAND ${CMAKE_COMMAND} --install
                ${_HEXAGON_BDIR}/${_P}-prefix/src/${_P}-build
        )
        list(APPEND _HTP_PREFIX_DIRS ${_HEXAGON_BDIR}/${_P}-prefix)
    endforeach()
    # Scrub the ExternalProject workspaces once the skels are copied into
    # ${_HEXAGON_BDIR} — inf2cat's recursive scan of /driver: must not see
    # .ninja_lock or other transient files under htp-v*-prefix/.
    list(APPEND _INSTALL_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${_HTP_PREFIX_DIRS}
    )
    add_custom_target(install-htp-skels
        ${_INSTALL_COMMANDS}
        DEPENDS ${_HTP_PROJECTS}
        COMMENT "Installing HTP skel libraries"
    )
    # inf2cat signing depends on the .so files being in ${_HEXAGON_BDIR} and
    # on the prefix dirs being gone. Both are now guaranteed by
    # install-htp-skels.
    if(TARGET libggml-htp-cat)
        add_dependencies(libggml-htp-cat install-htp-skels)
    endif()

    # Install HTP skels and catalog into pkg layout alongside ggml-hexagon.
    install(FILES ${_HTP_FILES} DESTINATION lib/llama_cpp)
    if(WIN32)
        install(FILES ${_HEXAGON_BDIR}/libggml-htp.cat DESTINATION lib/llama_cpp OPTIONAL)
    endif()

    unset(_HEXAGON_BDIR)
    unset(_P)
    unset(_HTP_PROJECTS)
    unset(_INSTALL_COMMANDS)
    unset(_HTP_PREFIX_DIRS)
    unset(_HTP_FILES)
endif()

# Adreno precompiled OpenCL kernel library (Software Center
# Adreno_Kernel_Library_GGML). ggml-opencl loads it by bare name and falls
# back to its builtin kernels when it is missing or has no kernel for the
# GPU; only X2 GPUs have matching kernels today. The plugin's DLL directory
# is lib/llama_cpp, so it installs next to ggml-opencl.dll. Pinned so a
# release is reproducible; Windows/latest.zip moves.
if(WIN32 AND GGML_OPENCL AND GGML_OPENCL_USE_ADRENO_BIN_KERNELS)
    set(_AK_VERSION 0.0.15)
    set(_AK_SHA256 325c89a3bce51abe95f40c6bacece005d5ca9cf6166eb5f17d33592cd13ab795)
    set(_AK_URL "https://softwarecenter.qualcomm.com/api/download/software/tools/Adreno_Kernel_Library_GGML/Windows/${_AK_VERSION}/adreno-opencl-kernels-${_AK_VERSION}.zip")
    set(_AK_DIR ${CMAKE_BINARY_DIR}/adreno-opencl-kernels-${_AK_VERSION})
    set(_AK_ZIP ${_AK_DIR}.zip)

    # Skips the download when the zip is already there with the pinned hash.
    file(DOWNLOAD ${_AK_URL} ${_AK_ZIP}
        EXPECTED_HASH SHA256=${_AK_SHA256}
        TLS_VERIFY ON
        STATUS _AK_STATUS)
    list(GET _AK_STATUS 0 _AK_CODE)
    if(NOT _AK_CODE EQUAL 0)
        message(FATAL_ERROR "Adreno kernel library download failed: ${_AK_STATUS}")
    endif()
    if(NOT EXISTS ${_AK_DIR}/adreno-opencl-kernels.dll)
        file(ARCHIVE_EXTRACT INPUT ${_AK_ZIP} DESTINATION ${_AK_DIR})
    endif()
    if(NOT EXISTS ${_AK_DIR}/adreno-opencl-kernels.dll)
        message(FATAL_ERROR "adreno-opencl-kernels.dll missing from ${_AK_ZIP}")
    endif()
    message(STATUS "Adreno kernel library ${_AK_VERSION}: ${_AK_DIR}")

    install(FILES ${_AK_DIR}/adreno-opencl-kernels.dll DESTINATION lib/llama_cpp)
    install(FILES ${_AK_DIR}/license.txt
        DESTINATION lib/llama_cpp
        RENAME adreno-opencl-kernels-license.txt)

    unset(_AK_VERSION)
    unset(_AK_SHA256)
    unset(_AK_URL)
    unset(_AK_DIR)
    unset(_AK_ZIP)
    unset(_AK_STATUS)
    unset(_AK_CODE)
endif()

# llama.cpp benefits from relaxed-FP (FMA contraction, reassociation)
if(WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    foreach(_t IN ITEMS llama ggml ggml-base ggml-cpu ggml-vulkan ggml-opencl ggml-hexagon mtmd)
        if(TARGET ${_t})
            target_compile_options(${_t} PRIVATE -ffp-model=fast -fno-finite-math-only)
        endif()
    endforeach()
    unset(_t)
endif()

# setup plugin

add_subdirectory(./src)

# setup install

set(LLAMA_LIBS llama-common mtmd llama ggml ggml-base ggml-cpu ggml-vulkan ggml-opencl ggml-hexagon)
foreach(dep IN LISTS LLAMA_LIBS)
    if(TARGET ${dep})
        # disable version numbers in so file names
        set_property(TARGET ${dep} PROPERTY VERSION)
        set_property(TARGET ${dep} PROPERTY SOVERSION)
        install(TARGETS ${dep}
            LIBRARY DESTINATION lib/llama_cpp
            RUNTIME DESTINATION lib/llama_cpp)
    endif()
endforeach()
