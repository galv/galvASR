if (__GALVASR_EXTERNAL_INCLUDED)
  return()
endif()
set(__GALVASR_EXTERNAL_INCLUDED TRUE)

include(ExternalProject)

set(caffe2_PREFIX ${PROJECT_SOURCE_DIR}/third_party/caffe2)
set(openfst_PREFIX ${PROJECT_SOURCE_DIR}/third_party/openfst)
set(kaldi_PREFIX ${PROJECT_SOURCE_DIR}/third_party/kaldi)
set(tensorflow_PREFIX ${PROJECT_SOURCE_DIR}/third_party/tensorflow)

# Required packages: Python, CUDA, Kaldi
if(NOT PYTHONINTERP_FOUND)
  # Note: Need to find PythonInterp before PythonLibs, as per documentation
  find_package(PythonInterp 3.5)
endif()
if(NOT PYTHONLIBS_FOUND)
  find_package(PythonLibs 3.5)
endif()

find_package(CUDA 9.0 QUIET)
if (CUDA_FOUND)
  message(STATUS "CUDA detected: " ${CUDA_VERSION})
endif()
find_package(CuDNN 7.0)

ExternalProject_Add(kaldi
  SOURCE_DIR ${kaldi_PREFIX}/kaldi/src
  BUILD_IN_SOURCE 1
  # CONFIGURE_COMMAND ""
  # TODO: Want to make configure arguments more customizable somehow,
  # at some point.
  CONFIGURE_COMMAND ./configure --shared
  BUILD_COMMAND $(MAKE) clean
  COMMAND $(MAKE) depend
  COMMAND $(MAKE)
  COMMAND $(MAKE) biglib
  INSTALL_COMMAND "")

ExternalProject_Add_Step(kaldi install_tools
  WORKING_DIRECTORY ${kaldi_PREFIX}/kaldi/tools
  # Trying adding CXX="-fPIC" and
  # OPENFST_CONFIGURE="--enable-static --enable-shared --enable-far --enable-ngram-fsts --enable-python" at some point
  COMMAND make
  COMMAND extras/install_irstlm.sh
  COMMAND extras/install_pocolm.sh
  DEPENDERS configure
  )

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
  ExternalProject_Add(tensorflow
    SOURCE_DIR ${tensorflow_PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=tensorflow_install
    BUILD_IN_SOURCE 1 # Not really true, but we may need this for Bazel build.
    CONFIGURE_COMMAND PYTHON_BIN_PATH=${PYTHON_EXECUTABLE}
                      USE_DEFAULT_PYTHON_LIB_PATH=1
                      TF_NEED_MKL=0
                      TF_NEED_JEMALLOC=0
                      TF_NEED_GCP=0
                      TF_NEED_HDFS=0
                      TF_NEED_S3=0
                      TF_ENABLE_XLA=0
                      TF_NEED_GDR=0
                      TF_NEED_VERBS=0
                      #TF_NEED_OPENCL_SYCL=${TF_NEED_OPENCL_SYCL}
                      TF_NEED_OPENCL=0
                      TF_NEED_MPI=0
                      TF_NEED_CUDA=1
                      TF_CUDA_CLANG=0
                      TF_CUDA_VERSION=${CUDA_VERSION}
                      CUDA_TOOLKIT_PATH=${CUDA_TOOLKIT_ROOT_DIR}
                      TF_CUDNN_VERSION=${CUDNN_VERSION}
                      CUDNN_INSTALL_PATH=${CUDNN_LIBRARIES}
                      TF_CUDA_CONFIG_REPO=""
                      # TODO: Make these the same as the compute
                      # capabilities used in Kaldi install
                      TF_CUDA_COMPUTE_CAPABILITIES=3.0,5.0,6.0
                      CC_OPT_FLAGS="-march=native"
                      TF_SET_ANDROID_WORKSPACE=""
                      ./configure
                      BUILD_COMMAND bazel build -c opt --config=cuda --config=monolithic //tensorflow:libtensorflow.so
                      COMMAND bazel build -c opt --config=cuda --config=monolithic //tensorflow:libtensorflow_cc.so
                      COMMAND bazel build -c opt --config=cuda --config=monolithic //tensorflow/tools/pip_package:build_pip_package
                      COMMAND bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PROJECT_BINARY_DIR}/tensorflow-prefix/tmp/tensorflow_pkg
                      # HACK: Find a way to expand the python wheel
                      # automatically in sucky cmake instead of
                      # hard-coding it!
                      INSTALL_COMMAND ${PYTHON_EXECUTABLE} -m pip install ${PROJECT_BINARY_DIR}/tensorflow-prefix/tmp/tensorflow_pkg/tensorflow-1.4.1-cp35-cp35m-linux_x86_64.whl)

  set(TENSORFLOW_INCLUDE_DIRS ${tensorflow_PREFIX}
    ${tensorflow_PREFIX}/bazel-genfiles
    ${tensorflow_PREFIX}/bazel-tensorflow/external/nsync/public
    ${tensorflow_PREFIX}/bazel-tensorflow/external/eigen_archive
    ${tensorflow_PREFIX}/bazel-tensorflow/external/protobuf_archive/src)
  set(TENSORFLOW_LIBRARIES
#    ${tensorflow_PREFIX}/bazel-out/local_linux-py3-opt/bin/tensorflow/libtensorflow.so
    ${tensorflow_PREFIX}/bazel-bin/tensorflow/libtensorflow_cc.so
#    ${tensorflow_PREFIX}/bazel-bin/tensorflow/libtensorflow_cc.so
)
                    
endif(USE_TENSORFLOW)
