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

# patch for hexagon signing after disable install
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

add_subdirectory(
    ../../../third-party/llama.cpp
    ${CMAKE_BINARY_DIR}/third-party/llama.cpp
    EXCLUDE_FROM_ALL)

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

set(LLAMA_LIBS mtmd llama ggml ggml-base ggml-cpu ggml-vulkan ggml-opencl ggml-hexagon)
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
