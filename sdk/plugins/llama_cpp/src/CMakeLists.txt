file(GLOB_RECURSE LLAMA_CPP_SRCS *.cpp)
add_library(geniex_llama_cpp SHARED ${LLAMA_CPP_SRCS})

target_compile_definitions(geniex_llama_cpp PUBLIC ML_SHARED)

target_include_directories(geniex_llama_cpp
    PUBLIC  ${COMMON_INCLUDE}
    PRIVATE ../include/)

target_link_libraries(geniex_llama_cpp
    PUBLIC  llama common
    PRIVATE geniex mtmd)

set_target_properties(geniex_llama_cpp PROPERTIES
    OUTPUT_NAME "geniex_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/llama_cpp")

foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set_target_properties(geniex_llama_cpp PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
        RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/llama_cpp"
        ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/llama_cpp")
endforeach()

if(UNIX)
    set_target_properties(geniex_llama_cpp PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
        BUILD_RPATH "$ORIGIN:$ORIGIN/.."
    )
endif()

if(TARGET install-htp-skels)
    add_dependencies(geniex_llama_cpp install-htp-skels)
endif()

install(TARGETS geniex_llama_cpp
    LIBRARY DESTINATION lib/llama_cpp
    RUNTIME DESTINATION lib/llama_cpp)
