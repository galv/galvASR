add_library(
  caffe2_ext SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_ark_input.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi_ark_input_gpu.cc)

target_link_libraries(caffe2_ext caffe2)
install(TARGETS caffe2_ext DESTINATION lib)
