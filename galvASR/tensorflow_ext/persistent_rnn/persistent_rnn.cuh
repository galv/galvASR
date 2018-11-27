extern "C"
__launch_bounds__(256, 1)
  __global__ void RNN_persist_fprop(const T_ELEM * __restrict__ x,
                                    T_GEMM_IN * __restrict__ y,
                                    const T_ELEM * __restrict__ hx,
                                    T_ELEM * __restrict__ hy,
                                    const T_ELEM * __restrict__ cx,
                                    T_ELEM * __restrict__ cy,
                                    T_ELEM * __restrict__ c_data, // LSTM-specific
                                    T_ELEM * __restrict__ tmp_h,
                                    T_ELEM * __restrict__ storedResults,
                                    const T_GEMM_IN * __restrict__ T,
                                    const T_ELEM * __restrict__ bias,
                                    const int seqLength) {
  const int THREADS_PER_BLOCK = WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y * 32;

  const int NUM_MATS = (RNN_MODE == CUDNN_LSTM) ? 4 :
    (RNN_MODE == CUDNN_GRU) ? 3 :
    1;

  const int BASIC_RNN = RNN_MODE == CUDNN_RNN_RELU || RNN_MODE == CUDNN_RNN_TANH;
  const int THREAD_Y_STRIDE = WARP_SIZE_Y;
  const int BLOCK_WRITE_LENGTH = WARPS_PER_BLOCK_X * WARP_SIZE_X * ELE_PER_THREAD_X / NUM_MATS;

  // Add padding of 1 to avoid bank conflicts
  const int SMEM_I_SIZE = ((BLOCK_WRITE_LENGTH * NUM_MATS) % 2 == 1) ?
    BLOCK_WRITE_LENGTH * NUM_MATS + 1 : BLOCK_WRITE_LENGTH * NUM_MATS;
  // Load input into shared memory for faster accesses?
  // GROUP_BATCH_SIZE purpose is not clear... Let's assume == 1 for now
  __shared__ T_MATH smemi[GROUP_BATCH_SIZE > 1 ? MINIBATCH : 2][SMEM_I_SIZE];
  // Double buffering?
  __shared__ T_MATH smemh[2][SMEM_I_SIZE];
  __shared__ T_MATH smemcx[BASIC_RNN ? 1 : BLOCK_WRITE_LENGTH][BASIC_RNN ? 1 : MINIBATCH];
  __shared__ T_MATH smembias[RNN_MODE == CUDUNN_GRU ? BLOCK_WRITE_LENGTH : 1];

  int warpIdBlock = threadIdx.x / 32;
  int warpIdGlobal = blockIdx.x * WARPS_PER_BLOCK_X * WARPS_PER_BLOCK_Y + warpIdBlock;
  int laneId = threadIdx.x % 32;

  int rowStartBlock;
  int rowStart;
  int colStart;

  rowStartBlock = ((warpIdBlock / WARPS_PER_BLOCK_Y) * WARP_SIZE_X +
                   (laneId % WARP_SIZE_X)) * ELE_PER_THREAD_X / NUM_MATS;
  rowStart = ((warpIdGlobal / WARPS_PER_BLOCK_Y) * WARP_SIZE_X +
              (laneId % WARP_SIZE_X)) * ELE_PER_THREAD_X / NUM_MATS;
  colStart = (laneId / WARP_SIZE_X) * INNER_UNROLL;

  colStart += (warpIdBlock % WARPS_PER_BLOCK_Y) * (VEC_LENGTH / WARPS_PER_BLOCK_Y);

  const int rowStride = (RNN_MODE == CUDNN_LSTM || RNN_MODE == CUDNN_GRU) ? HIDDEN_SIZE : 1;

  T_MATH T_reg[ELE_PER_THREAD_Y][ELE_PER_THREAD_X];

  RNN_persist_loadT<THREAD_Y_STRIDE, NUM_MATS, true>(T_reg, T, rowStart, colStart, rowStride);

  // What is the purpose of this zeroing?
  for (int i_ = 0; i_ < BLOCK_WRITE_LENGTH * NUM_MATS; i_ += THREADS_PER_BLOCK) {
    int i = i_ + threadIdx.x;
    if (i < BLOCK_WRITE_LENGTH * NUM_MATS) {
      smemi[0][i] = cuGet<T_MATH>(0);
      smemi[1][i] = cuGet<T_MATH>(0);
    }
  }
  for (int i_ = 0; i_ < WARPS_PER_BLOCK_Y * WARP_SIZE_Y * ELE_PER_THREAD_Y;
       i_ += THREADS_PER_BLOCK) {
    i = i_ + threadIdx.x;
    if (i < WARPS_PER_BLOCK_Y * WARP_SIZE_Y * ELE_PER_THREAD_Y) {
      smemh[0][i] = cuGet<T_MATH>(0);
      smemh[1][i] = cuGet<T_MATH>(0);
    }
  }

  if (RNN_MODE == CUDNN_LSTM || RNN_MODE == CUDNN_GRU) {
    for (int batch = 0; batch < MINIBATCH; batch++) {
      #pragma unroll
      for (int i_ = 0; i_ < BLOCK_WRITE_LENGTH; i_ += THREADS_PER_BLOCK) {
        int i = i_ + threadIdx.x;
        if (i < BLOCK_WRITE_LENGTH && i + BLOCK_WRITE_LENGTH * blockIdx.x < HIDDEN_SIZE) {
          if (RNN_MODE == CUDNN_LSTM) {
            if (cx != NULL) smemcx[i][batch] = cuGet<T_MATH>(cx[i + BLOCK_WRITE_LENGTH * blockIdx.x + batch * BATCH_STRIDE]);
            else            smemcx[i][batch] = cuGet<T_MATH>(0);
          }
        }
      }
    }
  }
}
                                    

__launch_bounds__(256, 1)
__global__ void RNN_persist_fprop(const T_ELEM * __restrict__ x,
                                    T_GEMM_IN * __restrict__ y,
                                    const T_ELEM * __restrict__ hx,
                                    T_ELEM * __restrict__ hy,
                                    const T_ELEM * __restrict__ cx,
                                    T_ELEM * __restrict__ cy,
                                    T_ELEM * __restrict__ c_data,
                                    )
