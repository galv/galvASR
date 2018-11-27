#pragma once

template<typename FloatT, class Context>
class SimpleCTCOp final : public Operator<Context> {
public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SimpleCTCOp(const OperatorDef& operator_def, Workspace* ws)
    : Operator<Context>(operator_def, ws),
    d_inclusive_scan_temp_size_(0),
    d_inclusive_scan_temp_storage_(nilptr),
    d_reduce_temp_size_(0),
    d_reduce_temp_storage_(nilptr)
    {
      CUDA_ENFORCE(cudaMalloc((void**)&num_repeats_, kMaxTranscriptLength));
    }
  ~SimpleCTCOp() {
    if (d_inclusive_scan_temp_storage_ != nilptr) {
      CUDA_ENFORCE(cudaFree(d_inclusive_scan_temp_storage_));
    }
    if (d_reduce_temp_storage_ != nilptr) {
      CUDA_ENFORCE(cudaFree(d_reduce_temp_storage_));
    }

    CUDA_ENFORCE(cudaFree(num_repeats_));
  }
  
  bool RunOnDevice() override;
private:
  INPUT_TAGS(PROBS, LABELS, LABEL_LENGTHS, INPUT_LENGTHS);
  OUTPUT_TAGS(NLL_FORWARD, NLL_BACKWARD, GRADIENT, ALPHA, BETA);

  size_t inclusive_scan_temp_size_;
  void  *d_inclusive_scan_temp_storage_;
  size_t reduce_temp_size_;
  void  *d_reduce_temp_storage_;

  static constexpr size_t kMaxTranscriptLength = 640;
  int *num_repeats_;
}
