add_definitions(${KALDI_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${CAFFE2_INCLUDE_DIRS})

add_library(
  caffe2_ext SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_ark_input.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_ark_input_gpu.cc)

target_link_libraries(caffe2_ext ${KALDI_LIBRARIES} ${CAFFE2_LIBRARIES})
install(TARGETS caffe2_ext DESTINATION lib)
