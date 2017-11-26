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

#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "kaldi_ark_input.h"

namespace galvASR {
REGISTER_CUDA_OPERATOR(KaldiFloatMatrixArchiveInput,
		       KaldiMatrixArchiveInputOp<kaldi::KaldiObjectHolder<kaldi::Matrix<float>>, CUDAContext>);

REGISTER_CUDA_OPERATOR(KaldiFloatVectorArchiveInput,
		       KaldiVectorArchiveInputOp<kaldi::KaldiObjectHolder<kaldi::Vector<float>>, CUDAContext>);

}
