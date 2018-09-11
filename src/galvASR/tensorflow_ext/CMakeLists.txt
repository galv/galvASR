# Ideally, we'd use this, but unfortunately we cannot link to
# libtensorflow_framework.so directly, since that would cause
# libtensorflow_framework.so to be loaded twice, which would
# cause ops to be registered twice.
#set(CMAKE_CXX_FLAGS "-Wl,--no-undefined")

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
