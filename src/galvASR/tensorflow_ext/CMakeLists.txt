set(CMAKE_CXX_FLAGS "-Wl,--no-undefined")
add_definitions(${KALDI_DEFINES} ${TENSORFLOW_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})

file(GLOB SRCS *.cc)

add_library(
  tensorflow_ext SHARED
  ${SRCS})
python_extension_module(tensorflow_ext)

# Try making this a shared library at some point.
add_library(
  tensorflow_ext_op_lib
  ${SRCS})

target_link_libraries(tensorflow_ext_op_lib ${KALDI_LIBRARIES})
#python_extension_module(tensorflow_ext_op_lib)
#target_link_libraries(tensorflow_ext_op_lib ${TENSORFLOW_LIBRARIES})
install(TARGETS tensorflow_ext LIBRARY DESTINATION "src/galvASR/python/tensorflow_ext")
# I just need to figure out how to generate _gen_dataset_ops.py
# automatically, with either CMake or Bazel. I think the following
# link may be enough:

# https://github.com/tensorflow/tensorflow/blob/e86d50aae5ea46a3d0f60286b5da91f2a4440101/tensorflow/contrib/cmake/tf_python.cmake

set_property(TARGET tensorflow_ext PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tensorflow_ext ${KALDI_LIBRARIES} ${TENSORFLOW_LIBRARIES})
target_link_libraries(tensorflow_ext Threads::Threads)
install(TARGETS tensorflow_ext LIBRARY DESTINATION "src/galvASR/python/tensorflow_ext")
