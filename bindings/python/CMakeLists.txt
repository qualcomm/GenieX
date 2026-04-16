find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PYTHON_DIST_DIR ${CMAKE_BINARY_DIR}/dist)
file(MAKE_DIRECTORY ${PYTHON_DIST_DIR})

string(REGEX REPLACE "^v" "" CLEAN_VERSION "${GENIEX_BRIDGE_VERSION}")

if(CLEAN_VERSION STREQUAL "latest")
    set(CLEAN_VERSION "0.0.0")
endif()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/geniex/_version.py.in
    ${CMAKE_CURRENT_SOURCE_DIR}/geniex/_version.py
    @ONLY
)

add_custom_target(
    python_wheel ALL
    COMMAND ${Python3_EXECUTABLE} -m build --sdist --outdir ${PYTHON_DIST_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building Python sdist to ${PYTHON_DIST_DIR}"
)

message(STATUS "Python binding: ${Python3_EXECUTABLE} -> ${PYTHON_DIST_DIR}")

