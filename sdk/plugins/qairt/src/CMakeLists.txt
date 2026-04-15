file(GLOB_RECURSE QAIRT_SRCS *.cpp)

if(WIN32)
    list(FILTER QAIRT_SRCS EXCLUDE REGEX "[/\\]vlm\\.cpp$")
endif()

add_library(geniex_qairt SHARED ${QAIRT_SRCS})

target_compile_features(geniex_qairt PRIVATE cxx_std_20)
target_compile_definitions(geniex_qairt PUBLIC ML_SHARED)
target_compile_definitions(geniex_qairt PRIVATE GENIEX_VERSION="${GENIEX_VERSION}")
set_target_properties(geniex_qairt PROPERTIES
    OUTPUT_NAME "geniex_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt")

foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set_target_properties(geniex_qairt PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt"
        RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt"
        ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt")
endforeach()

target_include_directories(geniex_qairt BEFORE
    PRIVATE ${GENIEX_QAIRT_DIR}/core/include
    PRIVATE ${GENIEX_QAIRT_DIR}/models/qwen3
    PRIVATE ${GENIEX_QAIRT_DIR}/models/phi4
    PRIVATE ${GENIEX_QAIRT_DIR}/models/phi3_5
    PRIVATE ${GENIEX_QAIRT_DIR}/models/granite4
    PRIVATE ${GENIEX_QAIRT_DIR}/models/qwen2-omni
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/include)
target_include_directories(geniex_qairt
    PUBLIC  ${COMMON_INCLUDE}
    PRIVATE ../include/)

set(QAIRT_PLUGIN_LINK_DEPS geniex geniex_core)
foreach(dep IN ITEMS geniex_vlm)
    if(TARGET ${dep})
        list(APPEND QAIRT_PLUGIN_LINK_DEPS ${dep})
    endif()
endforeach()

target_link_libraries(geniex_qairt PRIVATE ${QAIRT_PLUGIN_LINK_DEPS})

install(TARGETS geniex_qairt
    LIBRARY DESTINATION lib/qairt
    RUNTIME DESTINATION lib/qairt
    ARCHIVE DESTINATION lib/qairt)
