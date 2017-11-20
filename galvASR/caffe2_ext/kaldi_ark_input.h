/**
 * Copyright (c) 2017-present, Daniel Galvez
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/prefetch_op.h"

#include "kaldi/src/util/kaldi-holder.h"
#include "kaldi/src/util/kaldi-table.h"

namespace galvASR {

using namespace caffe2;

namespace {

template<typename KaldiType, typename Context>
struct KaldiToCaffe2Type {
  using T = KaldiType;
  using ContainedType = std::nullptr_t;
};

template<typename Context>
struct KaldiToCaffe2Type<kaldi::Matrix<float>, Context> {
  using T = caffe2::Tensor<Context>;
  using ContainedType = float;
  // using TensorType = float;
  // TypeMeta meta = TypeMeta::Id<float>();
};

template<typename Context>
struct KaldiToCaffe2Type<kaldi::Vector<float>, Context> {
  using T = caffe2::Tensor<Context>;
  using ContainedType = float;
  // using TensorType = float;
  // TypeMeta meta = TypeMeta::Id<float>();
};


}

template <typename Holder, typename Context>
class KaldiArkInputOp final : public PrefetchOperator<Context> {
 public:
  // USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool Prefetch() override;
  bool CopyPrefetched() override;
  explicit KaldiArkInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~KaldiArkInputOp() {
    PrefetchOperator<Context>::Finalize();
  }
 private:
  kaldi::SequentialTableReader<Holder> reader_;
  std::size_t batch_size_;
  std::vector<typename Holder::T> prefetched_data_;
  std::vector<std::string> prefetched_data_keys_;

  INPUT_TAGS(R_SPECIFIER);
};

template <typename Holder, typename Context>
KaldiArkInputOp<Holder, Context>::KaldiArkInputOp(const OperatorDef& operator_def,
                                                  Workspace* ws)
  : PrefetchOperator<Context>(operator_def, ws),
    reader_(OperatorBase::Input<std::string>(R_SPECIFIER)),
    OP_SINGLE_ARG(int, "batch_size", batch_size_, 1),
    prefetched_data_(batch_size_),
    prefetched_data_keys_(batch_size_) { }

template <typename Holder, typename Context>
bool KaldiArkInputOp<Holder, Context>::Prefetch() {
  for (std::size_t i = 0; i < batch_size_ && !reader_.Done(); ++i, reader_.Next()) {
    prefetched_data_[i] = reader_.Value();
    prefetched_data_keys_[i] = reader_.Key();
  }

  if (reader_.Done() && i != batch_size_ - 1) {
    return false;
  } else {
    return true;
  }
}

template <typename Holder, typename Context>
bool KaldiArkInputOp<Holder, Context>::CopyPrefetched() {
  using C2T = typename KaldiToCaffe2Type<typename Holder::T, Context>::T;
  using ContainedType = typename KaldiToCaffe2Type<typename Holder::T, Context>::ContainedType;

  for (std::size_t i = 0; i < OperatorBase::OutputSize(); ++i) {
    const std::vector<TIndex> dims = {prefetched_data_[i].NumRows(), prefetched_data_[i].NumCols()};
    C2T *tensor_to_fill = OperatorBase::Output<C2T>(i);
    tensor_to_fill->Resize(dims);
    this->context_.template Copy<ContainedType, CPUContext, Context>(
     tensor_to_fill->size(), prefetched_data_[i].Data(),
     tensor_to_fill->template mutable_data<ContainedType>());
  }
  return true;
}

} // namespace galvASR
