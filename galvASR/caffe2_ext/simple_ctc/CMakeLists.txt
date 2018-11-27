file(GLOB SRCS *.cc)
file(GLOB CUDA_SRCS *.cu)

include_directories(${CUB_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS "-Xptxas –v,abi=no ${NVCC_FLAGS}")
