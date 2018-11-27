#include <cmath>

#include "caffe2/core/context_gpu.h"
#include "cub/cub.cuh"

#include "simple_ctc_op.h"

namespace galvASR {
namespace {

template<typename T>
template<>
bool SimpleCTC<FloatT, CUDAContext>::RunOnDevice() {
  const auto& probs = Input(PROBS);
  const auto  minibatch_size = inputs.dim(1);
  const auto  alphabet_size  = inputs.dim(2);
  const auto& transcripts = ;
  const auto& transcript_lengths = ;
  const auto& inputLengths = ;
  Tensor<CUDAContext> *alpha = Output(ALPHA);

  transcript_offsets.ResizeLike(transcript_lengths);
  // TODO:Reduce to get max_transcript_length
  alpha->Resize(minibatch_size, max_input_length, max_transcript_length);
  cudaMemsetAsync(alpha->raw_mutable_data(), -std::numeric_limits<T>::infinity(),
                  alpha->nbytes(), this->context_.cuda_stream());

  // Could experiment with std::bind here...
  size_t new_exclusive_scan_temp_size = 0;
  cub::DeviceScan::ExclusiveSum(nilptr, new_exclusive_scan_temp_size,
                                transcript_lengths.template data<int>(),
                                transcript_offsets.template mutable_data<int>(),
                                transcript_lengths.dim32(0),
                                this->context_.cuda_stream());
  if (new_exclusive_scan_temp_size > inclusive_scan_temp_size_) {
    if (d_exclusive_scan_temp_storage_ != nilptr) {
      CUDA_ENFORCE(cudaFree(d_exclusive_scan_temp_storage_));
    }
    CUDA_ENFORCE(cudaMalloc(&d_exclusive_scan_temp_storage_, new_exclusive_scan_temp_size));
    exclusive_scan_temp_size_ = new_exclusive_scan_temp_size;
  }

  cub::DeviceScan::ExclusiveSum(d_inclusive_scan_temp_storage_,
                                new_inclusive_scan_temp_size,
                                transcript_lengths.template data<int>(),
                                transcript_offsets.template mutable_data<int>(),
                                transcript_lengths.dim32(0),
                                this->context_.cuda_stream());

  count_repeats<<<minibatch_size, kMaxTranscriptLength, 0, this->context_.cuda_stream()>>>(
    transcripts.template data<int>(),
    transcript_offsets.template data<int>(),
    transcript_lengths.template data<int>(),
    num_repeats_);

  
}

REGISTER_CUDA_OPERATOR(SimpleCTC, SimpleCTCOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(SimpleCTCGradient,
                       SimpleCTCGradientOp<float, CUDAContext>);
}
} // namespace galvASR
