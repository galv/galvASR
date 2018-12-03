#include "kaldi_variants.h"

#include "tensorflow/core/framework/op_kernel.h"

#include "kaldi/src/decoder/decodable-matrix.h"
#include "kaldi/src/decoder/simple-decoder.h"

#include "kaldi/src/matrix/matrix-lib.h"

namespace galvASR {

class SimpleDecoderOp final : public tensorflow::OpKernel {
public:
  explicit SimpleDecoderOp(tensorflow::OpKernelConstruction* c) :
    tensorflow::OpKernel(c) {
    LOG(INFO) << "SimpleDecoderOp constructor";
    OP_REQUIRES_OK(c, c->GetAttr("beam", &beam_));
    std::string HCLG_fst_rxfilename, tree_rxfilename, hmm_topo_rxfilename;
    OP_REQUIRES_OK(c, c->GetAttr("HCLG_fst", &HCLG_fst_rxfilename));
    OP_REQUIRES_OK(c, c->GetAttr("ctx_dep", &tree_rxfilename));
    OP_REQUIRES_OK(c, c->GetAttr("hmm_topo", &hmm_topo_rxfilename));

    {
      kaldi::ContextDependency ctx_dep;
      ReadKaldiObject(tree_rxfilename, &ctx_dep);
      kaldi::HmmTopology topo;
      ReadKaldiObject(hmm_topo_rxfilename, &topo);
      trans_model_.reset(new kaldi::TransitionModel(ctx_dep, topo));
    }

    hclg_.reset(fst::ReadFstKaldiGeneric(HCLG_fst_rxfilename));
  }

  void Compute(tensorflow::OpKernelContext* c) override {
    LOG(INFO) << "SimpleDecoderOp Compute";
    const tensorflow::Tensor& batched_input_tensor = c->input(0);
    // TODO: Use ThreadPool

    // TODO: Allow for log likelihoods of varying sizes. How to do this? Have an offsets vector?
    // Take a look at Mozilla Deep Speech code.
    tensorflow::Tensor *batched_output_ptr;
    c->allocate_output(0, tensorflow::TensorShape({batched_input_tensor.dim_size(0)}),
                       &batched_output_ptr);
    for (size_t i = 0; i < batched_input_tensor.dim_size(0); ++i) {
      // TODO: Make a decodable interface which operates on Tensor's
      // directly.
      tensorflow::Tensor input_tensor = batched_input_tensor.SubSlice(i);
      auto flat = input_tensor.flat<float>();
      kaldi::Matrix<float> input_matrix(input_tensor.dim_size(0),
                                        input_tensor.dim_size(1),
                                        kaldi::kUndefined,
                                        kaldi::kStrideEqualNumCols);
      std::copy_n(flat.data(), input_matrix.NumRows() * input_matrix.NumCols(),
                  input_matrix.Data());
      kaldi::DecodableMatrixScaledMapped decodable(*trans_model_, input_matrix, scale_);

      kaldi::SimpleDecoder decoder(*hclg_, beam_);
      decoder.Decode(&decodable);
      kaldi::Lattice *best_path = new kaldi::Lattice;
      decoder.GetBestPath(best_path);
      tensorflow::Variant best_path_var = LatticeVariant(best_path);
      batched_output_ptr->vec<float>()(i) = 0.0; //best_path_var;
      //batched_output_ptr->vec<tensorflow::Variant>()(i) = best_path_var;
    }
  }

private:
  // TODO: Acoustic or LM scale?
  float scale_;
  float beam_;
  std::unique_ptr<kaldi::TransitionModel> trans_model_;
  std::unique_ptr<fst::Fst<fst::StdArc>> hclg_;
};

REGISTER_KERNEL_BUILDER(Name("KaldiSimpleDecoder").
                        Device(tensorflow::DEVICE_CPU),
                        SimpleDecoderOp);

} // namespace galvASR
