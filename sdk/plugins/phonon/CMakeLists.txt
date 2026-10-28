# Phonon is an explicitly supplied, reviewed dependency. Never fetch or execute
# source code from a model bundle. See README.md for the tested source revision.
set(GENIEX_PHONON_SOURCE "" CACHE PATH "Gradium xn-ptts source with Windows QNN support")
if(NOT EXISTS "${GENIEX_PHONON_SOURCE}/qnn/runner/phonon.hpp")
    message(FATAL_ERROR "GENIEX_PHONON_SOURCE must point to the Gradium xn-ptts source tree")
endif()
find_package(Git REQUIRED)
set(_phonon_revision "9ee0eb5706d18ef1e401dc8b9d732c886089a5ab")
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GENIEX_PHONON_SOURCE}" rev-parse HEAD
    OUTPUT_VARIABLE _phonon_head OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _phonon_git_result)
if(NOT _phonon_git_result EQUAL 0 OR NOT _phonon_head STREQUAL _phonon_revision)
    message(FATAL_ERROR "Phonon requires reviewed xn-ptts revision ${_phonon_revision}; see README.md")
endif()
execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${GENIEX_PHONON_SOURCE}"
    status --porcelain --untracked-files=all -- qnn/runner
    OUTPUT_VARIABLE _phonon_dirty RESULT_VARIABLE _phonon_status_result)
if(NOT _phonon_status_result EQUAL 0 OR NOT _phonon_dirty STREQUAL "")
    message(FATAL_ERROR "The pinned Phonon qnn/runner source must be clean")
endif()
add_subdirectory("${GENIEX_PHONON_SOURCE}/qnn/runner" "${CMAKE_CURRENT_BINARY_DIR}/engine" EXCLUDE_FROM_ALL)
add_library(geniex_phonon SHARED plugin.cpp)
target_include_directories(geniex_phonon PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
target_link_libraries(geniex_phonon PRIVATE geniex phonon)
target_compile_definitions(geniex_phonon PRIVATE GENIEX_SHARED)
set_target_properties(geniex_phonon PROPERTIES OUTPUT_NAME geniex_plugin)
if(UNIX)
    set_target_properties(geniex_phonon PROPERTIES
        INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
        BUILD_RPATH "$ORIGIN:$ORIGIN/..")
endif()
install(TARGETS geniex_phonon LIBRARY DESTINATION lib/phonon_qnn RUNTIME DESTINATION lib/phonon_qnn)
install(FILES LICENSE-xn-ptts-MIT DESTINATION licenses/phonon_qnn)
include(FetchContent)
FetchContent_GetProperties(nlohmann_json)
install(FILES "${nlohmann_json_SOURCE_DIR}/LICENSE.MIT"
    DESTINATION licenses/phonon_qnn RENAME LICENSE-nlohmann-json-MIT)
