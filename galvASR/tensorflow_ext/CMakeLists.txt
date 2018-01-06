add_definitions(${KALDI_DEFINES})
include_directories(${KALDI_INCLUDE_DIRS} ${TENSORFLOW_INCLUDE_DIRS})

file(GLOB SRCS *.cc)

add_library(
  tensorflow_ext SHARED
  ${SRCS})

set_property(TARGET tensorflow_ext PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(tensorflow_ext ${KALDI_LIBRARIES})
# set_target_properties(tensorflow_ext PROPERTIES LIBRARY_OUTPUT_DIRECTORY
#   ${CMAKE_BINARY_DIR}/galvASR/python)
# install(TARGETS tensorflow_ext DESTINATION ${CMAKE_BINARY_DIR}/galvASR/python)
