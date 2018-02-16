add_definitions(${KALDI_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${CAFFE2_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})

file(GLOB SRCS *.cc)

add_library(
  caffe2_ext SHARED
  ${SRCS})

target_link_libraries(caffe2_ext ${KALDI_LIBRARIES} ${CAFFE2_LIBRARIES})
install(TARGETS caffe2_ext DESTINATION lib)
