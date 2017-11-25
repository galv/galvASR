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

#include "caffe2/core/module.h"

#include "kaldi_ark_input.h"

namespace galvASR {

CAFFE2_MODULE(kaldi_ark_input, "Read data in from Kaldi archives")

// Do we need to be in the caffe2 namespace to use this? That would be troublesome!
REGISTER_CPU_OPERATOR(KaldiFloatMatrixArchiveInput,
                      KaldiArkInputOp<kaldi::KaldiObjectHolder<kaldi::Matrix<float>>, CPUContext>);
SHOULD_NOT_DO_GRADIENT(KaldiFloatMatrixArchiveInput);
// REGISTER_CPU_OPERATOR(KaldiFloatVectorArchiveInput,
//                       KaldiArkInputOp<kaldi::KaldiObjectHolder<kaldi::Vector<float>>, CPUContext>);
// SHOULD_NOT_DO_GRADIENT(KaldiFloatVectorArchiveInput);
// REGISTER_CPU_OPERATOR(KaldiDoubleMatrixArchiveInput,
//                       KaldiArkInputOp<kaldi::Matrix<double>, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiDoubleVectorArchiveInput,
//                       KaldiArkInputOp<kaldi::Vector<double>, CPUContext>);

// Special case! wav files, woohoo!
// REGISTER_CPU_OPERATOR(KaldiDoubleVectorArchiveInput,
//                       KaldiArkInputOp<kaldi::Vector<double>, CPUContext>);

// Documentation will be nearly the same among all of these. Ugh! So ugly!
OPERATOR_SCHEMA(KaldiFloatMatrixArchiveInput)
  .NumInputs(1)
  .NumOutputs(1)
  .Arg("batch_size", "(int, default 1)");
// OPERATOR_SCHEMA(KaldiFloatVectorArchiveInput);
// OPERATOR_SCHEMA(KaldiDoubleMatrixArchiveInput);
// OPERATOR_SCHEMA(KaldiDoubleVectorArchiveInput);

// basic types
// REGISTER_CPU_OPERATOR(KaldiFloatArchiveInput,
//                       KaldiArkInputOp<float, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiDoubleArchiveInput,
//                       KaldiArkInputOp<double, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiBoolArchiveInput,
//                       KaldiArkInputOp<bool, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiIntArchiveInput,
//                       KaldiArkInputOp<int, CPUContext>);

// static_assert(sizeof(int) == 4, "Caffe2 assumes ints are 32-bit elsewhere. "
//               "This must hold true in order to be compatible with Kaldi's "
//               "architecture-independent int32 type.");

// // continers of basic types
// REGISTER_CPU_OPERATOR(KaldiIntVectorArchiveInput,
//                       KaldiArkInputOp<std::vector<int>, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiIntVectorVectorArchiveInput,
//                       KaldiArkInputOp<std::vector<std::vector<int>>, CPUContext>);
// REGISTER_CPU_OPERATOR(KaldiIntPairVectorArchiveInput,
//                       KaldiArkInputOp<std::vector<std::pair<int,int>>, CPUContext>);

}
