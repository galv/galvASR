#include <cublas_v2.h>

const float ONE = 1.0f;
const float ZERO = 0.0f;

// Test whether we can call cublasSgemmStridedBatched with strideC = 0
int main() {
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  int input_dim = 512;
  int output_dim = 512;
  int splice_stride = 2;
  int num_splices = 3;
  int sequence_length = num_splices * splice_stride;
  int batch_size = 64;


  float *A, *B, *C;
  CUDA_CALL(cudaMalloc(&A, batch_size * input_dim * sequence_length * sizeof *A));
  CUDA_CALL(cudaMalloc(&B, input_dim * output_dim * num_splices * sizeof *B));
  CUDA_CALL(cudaMalloc(&C, batch_size * (output_dim * num_splices) * sizeof *C));
  
  CUBLAS_CALL(cublasSgemmStridedBatched(handle,
                                        CUBLAS_OP_T, // I am using row-major matrices
                                        CUBLAS_OP_T,
                                        batch_size, // m
                                        output_dim, // n
                                        input_dim, // k
                                        &ONE,
                                        A,
                                        input_dim,
                                        batch_size * input_dim,
                                        B,
                                        output_dim,
                                        input_dim * output_dim,
                                        &ZERO,
                                        C,
                                        output_dim,
                                        0, // strideC
                                        num_splices
                                        ));
}