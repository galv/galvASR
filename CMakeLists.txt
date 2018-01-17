cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(galvASR CXX C)
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Add this back when we're ready for it, and when we can guarantee
# that it'll use the same python install that caffe2 uses.
# add_subdirectory(third_party/pybind11)
include(cmake/utils.cmake)
include(cmake/external.cmake)

# TODO: Make this depend on external dependencies already having beenb
# built.
add_subdirectory(galvASR)
