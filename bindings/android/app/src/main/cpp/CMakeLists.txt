project(android_project)

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use prebuilt SDK package at sdk/pkg-geniex
get_filename_component(PROJECT_ROOT "${CMAKE_SOURCE_DIR}/../../../../../.." ABSOLUTE)
set(SDK_INSTALL_DIR "${PROJECT_ROOT}/sdk/pkg-geniex")
set(geniex_bridge   "${SDK_INSTALL_DIR}/lib/libgeniex.so")
set(SDK_INCLUDE_DIR "${SDK_INSTALL_DIR}/include")

add_subdirectory(qnn)
