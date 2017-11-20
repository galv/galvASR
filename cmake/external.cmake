if (__GALVASR_EXTERNAL_INCLUDED)
  return()
endif()
set(__GALVASR_EXTERNAL_INCLUDED TRUE)

include(ExternalProject)

set(caffe2_PREFIX ${PROJECT_SOURCE_DIR}/third_party/caffe2)
set(openfst_PREFIX ${PROJECT_SOURCE_DIR}/third_party/openfst)
set(kaldi_PREFIX ${PROJECT_SOURCE_DIR}/third_party/kaldi)

ExternalProject_Add(caffe2
  SOURCE_DIR ${caffe2_PREFIX}
  # EP_BASE caffe2_install
  # PREFIX=caffe2_install
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=caffe2_install -DPYTHON_LIBRARY=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/Python -DPYTHON_INCLUDE_DIR=/usr/local/Cellar/python/2.7.14/Frameworks/Python.framework/Versions/2.7/Headers/
  # INSTALL_COMMAND cmake -E echo "Skipping install caffe2 step."
  # CONFIGURE_COMMAND ""
  # BUILD_COMMAND $(MAKE) -j
  # INSTALL_COMMAND ""
  )

set(CAFFE2_FOUND TRUE)
# Shouldn't caffe2 provide its .h files somehow? Maybe install command does that...
set(CAFFE2_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/caffe2-prefix/src/caffe2-build/caffe2_install/include/
  ${caffe2_PREFIX}/third_party/eigen/)
set(CAFFE2_LIBRARIES ${PROJECT_BINARY_DIR}/caffe2-prefix/src/caffe2-build/lib/libcaffe2.dylib)

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

set(OPENFST_FOUND TRUE)
set(OPENFST_INCLUDE_DIRS ${openfst_PREFIX}/include/)

set(KALDI_FOUND TRUE)
set(KALDI_DEFINES -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK)
set(KALDI_INCLUDE_DIRS ${kaldi_PREFIX} ${kaldi_PREFIX}/kaldi/src/ ${OPENFST_INCLUDE_DIRS})
# TODO: Understand why we need all directories, not just the first one...
#set(KALDI_LIBRARIES ${kaldi_PREFIX}/kaldi/src/lib/libkaldi.dylib)
file(GLOB KALDI_LIBRARIES ${kaldi_PREFIX}/kaldi/src/lib/libkaldi*.dylib)
