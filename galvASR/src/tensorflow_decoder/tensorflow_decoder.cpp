// Copyright 2017 Daniel Galvez

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "galvASR/tensorflow_decoder/tensorflow_decoder.hpp"

GALV_ASR_BEGIN_NAMESPACE

TensorFlowDecodable::TensorFlowDecodable(const tensorflow::GraphDef& graph,
                                         const tensorflow::SessionOptions& session_opts) :
    graph_(graph),
    session_()
{
    Session *out_session;
    TF_CHECK_OK(tensorflow::NewSession(session_opts, &out_session));
    session_ = std::unique_ptr(out_session);
}

float TensorFlowDecodable::LogLikelihood(int32_t frame, int32_t index) {
    session_->PRun();
}

GALV_ASR_END_NAMESPACE
