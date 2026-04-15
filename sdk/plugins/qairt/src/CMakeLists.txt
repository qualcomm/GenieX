file(GLOB_RECURSE QAIRT_SRCS *.cpp)
add_library(geniex_npu SHARED ${QAIRT_SRCS})

target_compile_definitions(geniex_npu PUBLIC ML_SHARED)
target_compile_definitions(geniex_npu PRIVATE GENIEX_VERSION="${GENIEX_VERSION}")
set_target_properties(geniex_npu PROPERTIES
    OUTPUT_NAME "geniex_plugin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/plugins/qairt")

foreach(cfg Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${cfg}" cfg_upper)
    set_target_properties(geniex_npu PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt"
        RUNTIME_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt"
        ARCHIVE_OUTPUT_DIRECTORY_${cfg_upper} "${CMAKE_BINARY_DIR}/plugins/qairt")
endforeach()

target_include_directories(geniex_npu
    PUBLIC  ${COMMON_INCLUDE}
    PRIVATE ../include/)

target_link_libraries(geniex_npu PRIVATE geniex geniex_core)

if(UNIX AND NOT ANDROID)
    target_link_libraries(geniex_npu PRIVATE
        -Wl,--no-as-needed nexaproc nexa-sampling -Wl,--as-needed
    )
endif()

install(TARGETS geniex_npu
    LIBRARY DESTINATION lib/qairt
    RUNTIME DESTINATION lib/qairt
    ARCHIVE DESTINATION lib/qairt)
