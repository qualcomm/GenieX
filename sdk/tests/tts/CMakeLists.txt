add_library(tts_mock SHARED mock.cpp)
target_include_directories(tts_mock PRIVATE ../../include)
target_link_libraries(tts_mock PRIVATE geniex)
target_compile_definitions(tts_mock PRIVATE GENIEX_SHARED)
set_target_properties(tts_mock PROPERTIES OUTPUT_NAME geniex_plugin
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/mock_tts"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/mock_tts"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/mock_tts")

add_library(tts_legacy SHARED mock.cpp)
target_include_directories(tts_legacy PRIVATE ../../include)
target_link_libraries(tts_legacy PRIVATE geniex)
target_compile_definitions(tts_legacy PRIVATE GENIEX_SHARED LEGACY_PLUGIN)
set_target_properties(tts_legacy PROPERTIES OUTPUT_NAME geniex_plugin
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/legacy"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/legacy"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/plugins/legacy")

add_executable(tts_contract contract.cpp c_header.c)
target_link_libraries(tts_contract PRIVATE geniex)
target_compile_definitions(tts_contract PRIVATE GENIEX_SHARED)
add_dependencies(tts_contract tts_mock tts_legacy)
add_custom_command(TARGET tts_contract POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:geniex> $<TARGET_FILE_DIR:tts_contract>)
# Multi-config generators place DLLs one level deeper; registry scans one level.
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/run-$<CONFIG>.cmake" CONTENT
    "file(MAKE_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}/scan-$<CONFIG>/mock_tts\" \"${CMAKE_CURRENT_BINARY_DIR}/scan-$<CONFIG>/legacy\")\nfile(COPY \"$<TARGET_FILE:tts_mock>\" DESTINATION \"${CMAKE_CURRENT_BINARY_DIR}/scan-$<CONFIG>/mock_tts\")\nfile(COPY \"$<TARGET_FILE:tts_legacy>\" DESTINATION \"${CMAKE_CURRENT_BINARY_DIR}/scan-$<CONFIG>/legacy\")\nexecute_process(COMMAND \"${CMAKE_COMMAND}\" -E env \"GENIEX_PLUGIN_PATH=${CMAKE_CURRENT_BINARY_DIR}/scan-$<CONFIG>\" \"$<TARGET_FILE:tts_contract>\" RESULT_VARIABLE result)\nif(NOT result EQUAL 0)\nmessage(FATAL_ERROR \"TTS contracts failed: \${result}\")\nendif()\n")
add_test(NAME tts_contract COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/run-$<CONFIG>.cmake")
set_tests_properties(tts_contract PROPERTIES TIMEOUT 30)
add_executable(tts_qnn_paths qnn_paths.cpp)
target_include_directories(tts_qnn_paths PRIVATE ../../include)
add_test(NAME tts_qnn_paths COMMAND tts_qnn_paths)
