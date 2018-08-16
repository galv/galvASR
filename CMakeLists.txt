cmake_minimum_required(VERSION 3.4.0)

project(galvASR CXX C)
set(CMAKE_CXX_STANDARD 11)

find_package(PythonExtensions REQUIRED)

set(PROJECT_TOP_DIR ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/cmake)

include(src/cmake/utils.cmake)
include(src/cmake/external.cmake)

add_subdirectory(src/galvASR)
