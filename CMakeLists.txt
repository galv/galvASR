cmake_minimum_required(VERSION 3.4.0)

project(galvASR CXX C)
set(CMAKE_CXX_STANDARD 11)

find_package(PythonExtensions REQUIRED)

set(PROJECT_TOP_DIR ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/galvASR/cmake)

include(galvASR/cmake/utils.cmake)
include(galvASR/cmake/external.cmake)

add_subdirectory(galvASR)
