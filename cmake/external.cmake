if (__GALVASR_EXTERNAL_INCLUDED)
  return()
endif()
set(__GALVASR_EXTERNAL_INCLUDED TRUE)

include(ExternalProject)

set(caffe2_PREFIX ${PROJECT_SOURCE_DIR}/third_party/caffe2)
set(kaldi_PREFIX ${PROJECT_SOURCE_DIR}/third_party/kaldi)

ExternalProject_Add(caffe2
  SOURCE_DIR ${caffe2_PREFIX}
  INSTALL_COMMAND cmake -E echo "Skipping install caffe2 step."
  # CONFIGURE_COMMAND ""
  # BUILD_COMMAND $(MAKE) -j
  # INSTALL_COMMAND ""
  )

ExternalProject_Add(kaldi
  SOURCE_DIR ${kaldi_PREFIX}/kaldi/src
  BUILD_IN_SOURCE 1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND $(MAKE)  -C ../tools/
  # TODO: Want to make configure arguments more customizable somehow,
  # at some point.
  COMMAND ./configure --shared
  COMMAND $(MAKE) clean
  COMMAND $(MAKE) depend
  COMMAND $(MAKE)
  COMMAND $(MAKE) biglib
  INSTALL_COMMAND "")

set(KALDI_FOUND TRUE)
set(KALDI_INCLUDE_DIRS ${kaldi_PREFIX})
set(KALDI_LIBARIES ${kaldi_PREFIX}/kaldi/src/)
