add_definitions(${KALDI_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${CAFFE2_INCLUDE_DIRS})

# TODO: Disable this when CUDA not available.
set(GALVASR_GPU_SRCS "")

file(GLOB tmp *.cc)
exclude(GALVASR_CPU_SRCS "${GALVASR_CPU_SRCS}" ${GALVASR_GPU_SRCS})

add_library(
  caffe2_ext SHARED
  ${GALVASR_CPU_SRCS}
  ${GALVASR_GPU_SRCS})

target_link_libraries(caffe2_ext ${KALDI_LIBRARIES} ${CAFFE2_LIBRARIES})
install(TARGETS caffe2_ext DESTINATION lib)
