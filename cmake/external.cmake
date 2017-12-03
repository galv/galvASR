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
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=caffe2_install
  # INSTALL_COMMAND cmake -E echo "Skipping install caffe2 step."
  # CONFIGURE_COMMAND ""
  # BUILD_COMMAND $(MAKE) -j
  # INSTALL_COMMAND ""
  )

set(CAFFE2_FOUND TRUE)
# Shouldn't caffe2 provide its .h files somehow? Maybe install command does that...
set(CAFFE2_INCLUDE_DIRS ${PROJECT_BINARY_DIR}/caffe2-prefix/src/caffe2-build/caffe2_install/include/
  ${caffe2_PREFIX}/third_party/eigen/)
file(GLOB CAFFE2_LIBRARIES
  ${PROJECT_BINARY_DIR}/caffe2-prefix/src/caffe2-build/lib/libcaffe2_gpu.so
  ${PROJECT_BINARY_DIR}/caffe2-prefix/src/caffe2-build/lib/libcaffe2.so)

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
file(GLOB OPENFST_LIBRARIES ${openfst_PREFIX}/lib/libfst*.so)

set(KALDI_FOUND TRUE)
set(KALDI_DEFINES -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK)
set(KALDI_INCLUDE_DIRS ${kaldi_PREFIX} ${kaldi_PREFIX}/kaldi/src/
  ${OPENFST_INCLUDE_DIRS}
  # Make Kaldi's blas includes happy.
  ${kaldi_PREFIX}/kaldi/tools/ATLAS_headers/include/ ${kaldi_PREFIX}/kaldi/tools/CLAPACK/)
# TODO: Understand why we need all directories, not just the first one...
file(GLOB KALDI_LIBRARIES ${kaldi_PREFIX}/kaldi/src/*/kaldi-*.a)
set(KALDI_LIBRARIES ${KALDI_LIBRARIES} ${OPENFST_LIBRARIES}
  # HACK: Would be better to query the required BLAS and LAPACK
  # libraries from Kaldi directly
  /usr/lib/libatlas.so.3 /usr/lib/libf77blas.so.3 /usr/lib/libcblas.so.3
  /usr/lib/liblapack_atlas.so.3
  )

find_package(CUDA 7.0 QUIET)
if (CUDA_FOUND)
  message(STATUS "CUDA detected: " ${CUDA_VERSION})
endif()
