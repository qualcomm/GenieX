add_library(geniex_qairt SHARED
    plugin.cpp llm.cpp vlm.cpp)

target_compile_features(geniex_qairt PRIVATE cxx_std_20)
target_compile_definitions(geniex_qairt PUBLIC GENIEX_SHARED)
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
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/include)
target_include_directories(geniex_qairt
    PUBLIC  ${COMMON_INCLUDE}
    PRIVATE ../include/
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/geniex-qairt/models)

target_link_libraries(geniex_qairt
    PRIVATE geniex geniex_core geniex-proc geniex_vlm)

install(TARGETS geniex_qairt
    LIBRARY DESTINATION lib/qairt
    RUNTIME DESTINATION lib/qairt)
