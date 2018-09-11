#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"

// namespace galvASR {
// namespace tensorflow_ext {

namespace tensorflow {

REGISTER_OP("KaldiFloat32MatrixDataset")
  .Input("r_specifier: string")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape);

REGISTER_OP("KaldiFloat32VectorDataset")
  .Input("r_specifier: string")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape)
  .Doc(R"doc(blah)doc");

REGISTER_OP("KaldiInt32VectorDataset")
  .Input("r_specifier: string")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape)
  .Doc(R"doc(blah)doc");

REGISTER_OP("KaldiWaveDataset")
  .Input("r_specifier: string")
  .Output("handle: variant")
  .SetIsStateful()
  .SetShapeFn(shape_inference::ScalarShape)
  .Doc(R"doc(blah)doc");

}

                                                \
// } // namespace tensorflow_ext
// } // namespace galvASR
