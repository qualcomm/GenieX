add_library(geniex_qairt SHARED
    plugin.cpp llm.cpp vlm.cpp log_filter.cpp)

target_compile_features(geniex_qairt PRIVATE cxx_std_20)
target_compile_definitions(geniex_qairt PUBLIC GENIEX_SHARED)
set_target_properties(geniex_qairt PROPERTIES
    OUTPUT_NAME "geniex_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt")

target_include_directories(geniex_qairt BEFORE
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/geniex-proc/include)
target_include_directories(geniex_qairt
    PUBLIC  ${COMMON_INCLUDE}
    PRIVATE ../include/
    PRIVATE ${CMAKE_SOURCE_DIR}/third-party/geniex-qairt/models)

target_link_libraries(geniex_qairt
    PRIVATE geniex geniex_core geniex-proc geniex_vlm)

# libQnnHtp is never unloaded, and its htp_ssr thread keeps calling the log
# callback registered through geniex_core into this plugin. Pin the plugin (and
# with it geniex_core) so Registry::clear()'s dlclose can't unmap that callback.
if(NOT WIN32)
    target_link_options(geniex_qairt PRIVATE -Wl,-z,nodelete)
endif()

install(TARGETS geniex_qairt
    LIBRARY DESTINATION lib/qairt
    RUNTIME DESTINATION lib/qairt)
