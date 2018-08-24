set(CMAKE_CXX_FLAGS "-Wl,--no-undefined")
add_definitions(${KALDI_DEFINES} ${TENSORFLOW_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})

file(GLOB SRCS *.cc)

add_library(
  tensorflow_ext SHARED
  ${SRCS})
python_extension_module(tensorflow_ext)
set_property(TARGET tensorflow_ext PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tensorflow_ext ${KALDI_LIBRARIES} ${TENSORFLOW_LIBRARIES})
target_link_libraries(tensorflow_ext Threads::Threads)
install(TARGETS tensorflow_ext LIBRARY DESTINATION "src/galvASR/python/tensorflow_ext")

# Try making this a shared library at some point.
add_library(
  tensorflow_ext_op_lib
  kaldi_dataset_ops.cc)

# We could generate the BUILD file right here. And make a custom
# target of gen_kaldi_table_dataset.py. That sounds pretty okay.

add_custom_command(OUTPUT gen_kaldi_table_dataset.py
  COMMAND bazel build --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --host_cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --verbose_failures -s --define grpc_no_ares=true --action_env OP_LIB="${CMAKE_CURRENT_BINARY_DIR}/libtensorflow_ext_op_lib.a"  //src/galvASR/python/tensorflow_ext:kaldi_table_dataset
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  DEPENDS tensorflow_ext_op_lib
  )

add_custom_target(dummy_target ALL
  DEPENDS gen_kaldi_table_dataset.py
  )

