cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)

include(ExternalProject)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/..)
set(tensorflow_PREFIX ${PROJECT_SOURCE_DIR}/../../third_party/tensorflow)

find_package(CUDA 9.0 QUIET)
if (CUDA_FOUND)
  message(STATUS "CUDA detected: ${CUDA_VERSION}")
  message(STATUS "CUDA toolkit location: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()
find_package(CuDNN 7.0)

if(NOT PYTHONINTERP_FOUND)
  # Note: Need to find PythonInterp before PythonLibs, as per documentation
  find_package(PythonInterp 3.5)
endif()

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
                      TF_NEED_KAFKA=0
                      TF_ENABLE_XLA=0
                      TF_NEED_GDR=0
                      TF_NEED_VERBS=0
                      TF_NEED_OPENCL_SYCL=0
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
                      TF_CUDA_COMPUTE_CAPABILITIES=6.1
                      CC_OPT_FLAGS="-march=native"
                      TF_SET_ANDROID_WORKSPACE=0
                      ./configure
                      BUILD_COMMAND bazel build -c opt --config=monolithic --config=cuda //tensorflow/tools/pip_package:build_pip_package
                      COMMAND bazel-bin/tensorflow/tools/pip_package/build_pip_package ${PROJECT_BINARY_DIR}/tensorflow-prefix/tmp/tensorflow_pkg
                      # HACK: Find a way to expand the python wheel
                      # automatically in sucky cmake instead of
                      # hard-coding it!
                      # INSTALL_COMMAND ${PYTHON_EXECUTABLE} -m pip install --ignore-installed ${PROJECT_BINARY_DIR}/tensorflow-prefix/tmp/tensorflow_pkg/tensorflow-1.6.0rc0-cp35-cp35m-linux_x86_64.whl
                      )
