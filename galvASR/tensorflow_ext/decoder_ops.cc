#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace galvASR {

REGISTER_OP("KaldiSimpleDecoder")
  .Input("log_likelihoods: float")
  .Attr("HCLG_fst: string")
  .Attr("ctx_dep: string")
  .Attr("hmm_topo: string")
  .Attr("beam: float")
  .Attr("scale: float")
  .Output("best_path: float")
//  .Output("best_path: variant")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
                LOG(INFO) << "Daniel galvez logging in shape inference!";
    ::tensorflow::shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &input_shape));
    // log_likelihoods is B x T x F right now, but in RNNs we should
    // expect T x B x F. How to differentiate that in the shape
    // inference function?
    ::tensorflow::shape_inference::DimensionHandle batch_size =
        c->Dim(input_shape, 0);
    c->set_output(0,c->Matrix(batch_size, c->kUnknownDim));
  });

} // namespace galvASR
