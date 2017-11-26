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
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/prefetch_op.h"

#include "kaldi/src/util/kaldi-holder.h"
#include "kaldi/src/util/kaldi-table.h"

namespace galvASR {

using namespace caffe2;

namespace detail {

template<typename KaldiType>
struct ContainedType {
  using T = void;
};

template<>
struct ContainedType<kaldi::Matrix<float>> {
  using T = float;
};

template<>
struct ContainedType<kaldi::Vector<float>> {
  using T = float;
};
}

template <typename Holder, typename Context>
class KaldiMatrixArchiveInputOp final : public PrefetchOperator<Context> {
 public:
  using MatrixType = typename Holder::T;
  bool Prefetch() override;
  bool CopyPrefetched() override;
  explicit KaldiMatrixArchiveInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~KaldiMatrixArchiveInputOp() {
    PrefetchOperator<Context>::Finalize();
  }
 private:
  kaldi::SequentialTableReader<Holder> reader_;
  std::size_t batch_size_;
  std::vector<MatrixType> prefetched_data_;
  std::vector<std::string> prefetched_data_keys_;

  INPUT_TAGS(R_SPECIFIER);
};

template <typename Holder, typename Context>
KaldiMatrixArchiveInputOp<Holder, Context>::KaldiMatrixArchiveInputOp(const OperatorDef& operator_def,
                                                  Workspace* ws)
  : PrefetchOperator<Context>(operator_def, ws),
    reader_(OperatorBase::Input<std::string>(R_SPECIFIER)),
    OP_SINGLE_ARG(int, "batch_size", batch_size_, 1),
    prefetched_data_(batch_size_),
    prefetched_data_keys_(batch_size_) { }

template <typename Holder, typename Context>
bool KaldiMatrixArchiveInputOp<Holder, Context>::Prefetch() {
  std::size_t i;
  for (i = 0; i < batch_size_ && !reader_.Done(); ++i, reader_.Next()) {
    // Remove any padding in the rows of the matrix, since Caffe2
    // tensors have no padding (see CopyPrefetched() to see how copying is done).
    // Value() returns a const reference, so we need to copy it in
    // order to remove padding.
    MatrixType data = reader_.Value();
    data.Resize(data.NumRows(), data.NumCols(), kaldi::kCopyData,
		kaldi::kStrideEqualNumCols);
    // TODO: Add move semantics to Kaldi matrices.
    prefetched_data_[i].Swap(&data);
    prefetched_data_keys_[i] = reader_.Key();
  }

  // May want to reconsider this...
  if (reader_.Done() && i != batch_size_ - 1) {
    return false;
  } else {
    return true;
  }
}

template <typename Holder, typename Context>
bool KaldiMatrixArchiveInputOp<Holder, Context>::CopyPrefetched() {
  using ContainedType = typename detail::ContainedType<MatrixType>::T;

  for (std::size_t i = 0; i < batch_size_; ++i) {
    MatrixType& data = prefetched_data_[i];
    data.Resize(data.NumRows(), data.NumCols(), kaldi::kCopyData,
		kaldi::kStrideEqualNumCols);
    CAFFE_ENFORCE_EQ(data.NumCols(), data.Stride(),
		     "Implementation error in kaldi::Matrix::Resize");
    const std::vector<TIndex> dims = {data.NumRows(), data.NumCols()};
    Tensor<Context> *tensor_to_fill = OperatorBase::Output<Tensor<Context>>(0);
    tensor_to_fill->Resize(dims);
    this->context_.template Copy<ContainedType, CPUContext, Context>(
      tensor_to_fill->size(), data.Data(),
      tensor_to_fill->template mutable_data<ContainedType>());
  }

  // TODO: Can I use move semantics here to avoid copying the keys?
  // *OperatorBase::Output<std::vector<unsigned long>>(1) = {1,2,3}; // std::move(prefetched_data_keys_);
  *OperatorBase::Output<std::vector<unsigned long>>(1) = {};
  
  return true;
}

template <typename Holder, typename Context>
class KaldiVectorArchiveInputOp final : public PrefetchOperator<Context> {
 public:
  using VectorType = typename Holder::T;
  bool Prefetch() override;
  bool CopyPrefetched() override;
  explicit KaldiVectorArchiveInputOp(const OperatorDef& operator_def, Workspace* ws);
  ~KaldiVectorArchiveInputOp() {
    PrefetchOperator<Context>::Finalize();
  }
 private:
  kaldi::SequentialTableReader<Holder> reader_;
  std::size_t batch_size_;
  std::vector<VectorType> prefetched_data_;
  std::vector<std::string> prefetched_data_keys_;

  INPUT_TAGS(R_SPECIFIER);
};
template <typename Holder, typename Context>
KaldiVectorArchiveInputOp<Holder, Context>::
KaldiVectorArchiveInputOp(const OperatorDef& operator_def, Workspace* ws)
  : PrefetchOperator<Context>(operator_def, ws),
  reader_(OperatorBase::Input<std::string>(R_SPECIFIER)),
  OP_SINGLE_ARG(int, "batch_size", batch_size_, 1),
  prefetched_data_(batch_size_),
  prefetched_data_keys_(batch_size_) { }

template <typename Holder, typename Context>
bool KaldiVectorArchiveInputOp<Holder, Context>::Prefetch() {
  std::size_t i;
  for (i = 0; i < batch_size_ && !reader_.Done(); ++i, reader_.Next()) {
    VectorType data = reader_.Value();
    prefetched_data_[i].Swap(&data);
    prefetched_data_keys_[i] = reader_.Key();
  }

  if (reader_.Done() && i != batch_size_ - 1) {
    return false;
  } else {
    return true;
  }
}

template <typename Holder, typename Context>
bool KaldiVectorArchiveInputOp<Holder, Context>::CopyPrefetched() {
  using ContainedType = typename detail::ContainedType<VectorType>::T;

  for (std::size_t i = 0; i < batch_size_; ++i) {
    VectorType& data = prefetched_data_[i];
    const std::vector<TIndex> dims = {data.Dim()};
    Tensor<Context> *tensor_to_fill = OperatorBase::Output<Tensor<Context>>(0);
    tensor_to_fill->Resize(dims);
    this->context_.template Copy<ContainedType, CPUContext, Context>(
      tensor_to_fill->size(), data.Data(),
      tensor_to_fill->template mutable_data<ContainedType>());
  }

  // TODO: Can I use move semantics here to avoid copying the keys?
  // *OperatorBase::Output<std::vector<unsigned long>>(1) = {1,2,3}; // std::move(prefetched_data_keys_);
  *OperatorBase::Output<std::vector<unsigned long>>(1) = {};
  
  return true;
}

} // namespace galvASR
