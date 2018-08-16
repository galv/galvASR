if (__GALVASR_EXTERNAL_INCLUDED)
  return()
endif()
set(__GALVASR_EXTERNAL_INCLUDED TRUE)

include(ExternalProject)

if(DEFINED WITH_EXTERNAL_KALDI)
  get_filename_component(kaldi_PREFIX "${WITH_EXTERNAL_KALDI}" ABSOLUTE)
else(DEFINED WITH_EXTERNAL_KALDI)
  set(kaldi_PREFIX ${PROJECT_TOP_DIR}/third_party/kaldi/kaldi)
endif(DEFINED WITH_EXTERNAL_KALDI)

set(openfst_PREFIX ${PROJECT_TOP_DIR}/third_party/openfst)
set(caffe2_PREFIX ${PROJECT_TOP_DIR}/third_party/caffe2)
set(tensorflow_PREFIX ${PROJECT_TOP_DIR}/third_party/tensorflow)

# Required packages: Python, CUDA, Kaldi
if(NOT PYTHONINTERP_FOUND)
  # Note: Need to find PythonInterp before PythonLibs, as per documentation
  find_package(PythonInterp 3.5)
endif()
if(NOT PYTHONLIBS_FOUND)
  find_package(PythonLibs 3.5)
endif()

# Python site-packages
# Get canonical directory for python site packages (relative to install
# location).  It varys from system to system.
pycmd(python_site_packages "
       from distutils import sysconfig
       print(sysconfig.get_python_lib(prefix=''))
   ")

# We need to install pywrapfst.so ourselves manually.
pycmd(full_python_site_packages "
       from distutils import sysconfig
       print(sysconfig.get_python_lib())
   ")


find_package(CUDA 9.0 QUIET)
if (CUDA_FOUND)
  message(STATUS "CUDA detected: ${CUDA_VERSION}")
  message(STATUS "CUDA toolkit location: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()
find_package(CuDNN 7.0)

if(NOT DEFINED WITH_EXTERNAL_KALDI)
  ExternalProject_Add(kaldi
    SOURCE_DIR ${kaldi_PREFIX}/src
    BUILD_IN_SOURCE 1
    # TODO: Want to make configure arguments more customizable somehow,
    # at some point.
    CONFIGURE_COMMAND ./configure --static-fst --shared --cudatk-dir=${CUDA_TOOLKIT_ROOT_DIR}
    # BUILD_COMMAND make clean
    BUILD_COMMAND make depend
    COMMAND make
    COMMAND make biglib
    INSTALL_COMMAND "")

  ExternalProject_Add_Step(kaldi install_tools
    WORKING_DIRECTORY ${kaldi_PREFIX}/tools
    # COMMAND make clean || exit 0
    COMMAND PYTHON=${PYTHON_EXECUTABLE} make OPENFST_CONFIGURE=--enable-static\ --enable-shared\ --enable-far\ --enable-ngram-fsts\ --enable-python CXXFLAGS=-fPIC
    COMMAND extras/install_irstlm.sh
    COMMAND extras/install_pocolm.sh
    DEPENDERS configure
    )
endif(NOT DEFINED WITH_EXTERNAL_KALDI)

set(OPENFST_FOUND TRUE)
set(OPENFST_INCLUDE_DIRS ${openfst_PREFIX}/include/)
file(GLOB OPENFST_LIBRARIES ${openfst_PREFIX}/lib/libfst*.a)

# We should really install this regardless of whether galvASR is being
# installed. Best to factor openfst out into its own
# "ExternalProject_Add" separate from Kaldi.
file(GLOB PYWRAPFST_SO "${kaldi_PREFIX}/tools/openfst/lib/python3*/site-packages/pywrapfst.so")
# install(FILES "${PYWRAPFST_SO}"
#   DESTINATION "${full_python_site_packages}")

set(KALDI_FOUND TRUE)
set(KALDI_DEFINES -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 -DHAVE_CXXABI_H -DHAVE_CLAPACK)
# See third_party/kaldi/README to understand the first include directory here.
set(KALDI_INCLUDE_DIRS ${kaldi_PREFIX}/.. ${kaldi_PREFIX}/src/
  ${OPENFST_INCLUDE_DIRS}
  # Make Kaldi's blas includes happy.
  ${kaldi_PREFIX}/tools/ATLAS_headers/include/ ${kaldi_PREFIX}/tools/CLAPACK/)
# TODO: Understand why we need all directories, not just the first one...
file(GLOB KALDI_LIBRARIES ${kaldi_PREFIX}/src/*/kaldi-*.a)
# Hack: Globbing does not get the archives in dependency order, and
# I'm not sure how to get them into dependency order, so list them all
# twice.
set(KALDI_LIBRARIES ${KALDI_LIBRARIES} ${KALDI_LIBRARIES})
# set(KALDI_LIBRARIES "${kaldi_PREFIX}/src/lib/libkaldi.so" "-Wl,-rpath,${kaldi_PREFIX}/src/lib/")
set(KALDI_LIBRARIES ${KALDI_LIBRARIES} ${OPENFST_LIBRARIES}
  # HACK: Would be better to query the required BLAS and LAPACK
  # libraries from Kaldi directly
  /usr/lib/libatlas.so.3 /usr/lib/libf77blas.so.3 /usr/lib/libcblas.so.3
  /usr/lib/liblapack_atlas.so.3
  )

# Optional packages: Caffe2, Tensorflow
if(USE_CAFFE2) 
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
endif(USE_CAFFE2)

message("CUDNN ${CUDNN_VERSION}")
message("CUDNN ${CUDNN_LIBRARIES}")
message("CUDNN ${CUDNN_LIBRARY_DIRS}")

if(USE_TENSORFLOW)
  pycmd(TENSORFLOW_INCLUDE_DIRS "
       import tensorflow as tf
       print(' '.join([flag[2:] for flag in tf.sysconfig.get_compile_flags() if flag.startswith('-I')]))
  ")
  separate_arguments(TENSORFLOW_INCLUDE_DIRS)

  pycmd(TENSORFLOW_DEFINES "
       import tensorflow as tf
       print(' '.join([flag for flag in tf.sysconfig.get_compile_flags() if flag.startswith('-D')]))
  ")
  separate_arguments(TENSORFLOW_DEFINES)

  pycmd(TENSORFLOW_LIBRARIES "
       import tensorflow as tf
       print(' '.join(tf.sysconfig.get_link_flags()))
  ")
  separate_arguments(TENSORFLOW_LIBRARIES)
endif(USE_TENSORFLOW)

if(DEFINED WITH_HTK)
  # Assumes HTK is pre-installed
  if(NOT EXISTS "${WITH_HTK}/bin/HInit")
    message(FATAL_ERROR "${WITH_HTK} is not pointing to an HTK installation directory!")
  endif()
  message(WARNING "The HTK License prohibits re-distribution of HTK code in "
    "commercial applications.")
  # Used in galvASR/python/_constants.py.in
  get_filename_component(htk_PREFIX "${WITH_HTK}" ABSOLUTE)
endif(DEFINED WITH_HTK)

# Kaldi depends on pthreads
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
