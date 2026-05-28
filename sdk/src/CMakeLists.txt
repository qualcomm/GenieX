file(GLOB BRIDGE_SRC *.cpp)

add_library(geniex SHARED ${BRIDGE_SRC})

set_source_files_properties(version.cpp PROPERTIES
    COMPILE_DEFINITIONS GENIEX_VERSION="${GENIEX_VERSION}")
target_compile_definitions(geniex PRIVATE GENIEX_SHARED GENIEX_BUILD)

target_include_directories(geniex PUBLIC ${COMMON_INCLUDE})

if(GENIEX_MODEL_MANAGER)
    add_dependencies(geniex model_manager_build)

    # The `RUST_MODEL_LIB` variable (set by sdk/CMakeLists.txt) points
    # at either libgeniex_model.a (Unix) or geniex_model.lib (Windows).
    if(WIN32)
        # On Windows we cannot use /WHOLEARCHIVE against geniex_model.lib:
        # cargo's staticlib output embeds Microsoft import-library stubs
        # (kernel32.dll, ws2_32.dll, ...) inside the archive, and
        # /WHOLEARCHIVE visits those members as if they were real object
        # files — lld-link rejects them with
        #   .idata$4 should not refer to special section 0
        #
        # We use /INCLUDE: to force the linker to pull in every Rust
        # FFI entry point, and /EXPORT: to publish them on libgeniex.dll.
        # Everything else in the staticlib stays local.
        set(GENIEX_MODEL_FFI_SYMBOLS
            geniex_model_init
            geniex_model_deinit
            geniex_model_pull
            geniex_model_list
            geniex_model_list_free
            geniex_model_remove
            geniex_model_clean
            geniex_model_get_paths
            geniex_model_paths_free
            geniex_model_get_type
            geniex_model_resolve_alias)
        target_link_libraries(geniex PRIVATE "${RUST_MODEL_LIB}")
        foreach(_sym ${GENIEX_MODEL_FFI_SYMBOLS})
            target_link_options(geniex PRIVATE
                "LINKER:/INCLUDE:${_sym}"
                "LINKER:/EXPORT:${_sym}")
        endforeach()
        # Rust's Windows runtime pulls in a handful of system libraries
        # (networking, crypto, userenv, ntdll helpers). Link them
        # explicitly so the loader can resolve every Rust reference.
        target_link_libraries(geniex PRIVATE
            ntdll userenv bcrypt crypt32 secur32 ncrypt ws2_32)
    else()
        target_link_libraries(geniex PRIVATE
            -Wl,--whole-archive
            "${RUST_MODEL_LIB}"
            -Wl,--no-whole-archive
        )
        # Restrict the dynamic-symbol table to the geniex_* API so the
        # Rust runtime / hf-hub / ring / ureq symbols that --whole-archive
        # pulls in do not leak into libgeniex.so's public ABI.
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Android")
            set(GENIEX_MODEL_VERSION_SCRIPT
                "${CMAKE_SOURCE_DIR}/model-manager/include/geniex_model.ld")
            target_link_options(geniex PRIVATE
                "LINKER:--version-script=${GENIEX_MODEL_VERSION_SCRIPT}")
            set_property(TARGET geniex APPEND PROPERTY
                LINK_DEPENDS "${GENIEX_MODEL_VERSION_SCRIPT}")
        endif()
    endif()
    target_include_directories(geniex PUBLIC
        "${CMAKE_SOURCE_DIR}/model-manager/include"
    )
    install(FILES "${CMAKE_SOURCE_DIR}/model-manager/include/geniex_model.h"
        DESTINATION include)

    if(GENIEX_BENCHMARK)
        add_executable(test_model_manager
            "${CMAKE_SOURCE_DIR}/model-manager/tests/test_model_manager.c"
        )
        target_include_directories(test_model_manager PRIVATE
            ${COMMON_INCLUDE}
            "${CMAKE_SOURCE_DIR}/model-manager/include"
        )
        target_link_libraries(test_model_manager PRIVATE geniex)
        add_test(NAME test_model_manager COMMAND test_model_manager)
    endif()
endif()

install(TARGETS geniex
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib)

install(FILES ${CMAKE_SOURCE_DIR}/include/geniex.h DESTINATION include)
