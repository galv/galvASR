#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace galvASR {

template <class Holder, class Context>
class KaldiArkInputOp final : public caffe2::PrefetchOperator<Context> {
 public:
  bool Prefetch() override;
  bool CopyPrefetched() override;
  explicit KaldiArkInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~KaldiArkInputOp() {
    PrefetchOperator<Context>::Finalize();
  }
 private:
  kaldi::RandomAccessTableReader<Holder> reader_;
  // May want to define a template struct to select which Caffe2 type
  // to output here... Nothing about kaldi tables guarantees values
  // for different keys are the same size for vectors and matrices!
  // Yikes!
  std::vector<Holder::T> prefetched_data_;
  int batch_size_;
  std::vector<std::string> prefetched_data_keys_;

  INPUT_TAGS(R_SPECIFIER);
};

template <class Holder, class Context>
KaldiArkInputOp<Holder, Context>::KaldiArkInputOp(const OperatorDef& operator_def,
                                                  Workspace* ws)
  : OperatorBase(operator_def, ws),
    reader_(OperatorBase::Input<std::string>(R_SPECIFIER)),
    OP_SINGLE_ARG(int, "batch_size", batch_size_, 1) { }

template <class Holder, class Context>
bool KaldiArkInputOp<Holder, Context>::Prefetch() {
  
}

template <class Holder, class Context>
KaldiArkInputOp<Holder, Context>::CopyPrefetched() {
  
}

} // namespace galvASR
