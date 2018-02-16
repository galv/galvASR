namespace galvASR {
namespace {

}
} // namespace galvASR
OPERATOR_SCHEMA(SimpleCTC)
  .NumInputs(3)
  .NumOutputs(2)
  .TensorInferenceFunction(...TODO...)
  .SetDoc(R"DOC(
)DOC")
  // Does the label_size really need to be an input?
//.Arg("label_size", "(int32_t) Total number of labels in the vocabulary.") - Derivable from probs
  .Arg("blank_label", "(int32_t) ")
  // Probably best to enforce that all inputs have the same length T...
  .Input(0, "probs", "3d input of size (T x N x P), where T is the max length "
         "of all, N is the minibatch size, and P is equal to label_size")
  .Input(1, "labels", "2d CSR style matrix")
  .Input(2, "label_lengths", "")
  .Input(3, "input_lengths", "")
  .Output(0, "nll_forward", "")
  // Is this possible?
  .Output(1, "nll_backward", "")
  .Output(2, "gradient", "")
  .Output(3, "alpha", "N x T x P")
  .Output(4, "beta", "N x T x P");
