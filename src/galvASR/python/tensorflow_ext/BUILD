load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

cc_import(name = "tensorflow_ext_op_lib",
          static_library = "libtensorflow_ext_op_lib.a",
          alwayslink=1)

# third_party/kaldi/kaldi/src/base/kaldi-base.a              third_party/kaldi/kaldi/src/ivector/kaldi-ivector.a  third_party/kaldi/kaldi/src/online2/kaldi-online2.a
# third_party/kaldi/kaldi/src/chain/kaldi-chain.a            third_party/kaldi/kaldi/src/kws/kaldi-kws.a          third_party/kaldi/kaldi/src/rnnlm/kaldi-rnnlm.a
# third_party/kaldi/kaldi/src/cudamatrix/kaldi-cudamatrix.a  third_party/kaldi/kaldi/src/lat/kaldi-lat.a          third_party/kaldi/kaldi/src/sgmm2/kaldi-sgmm2.a
# third_party/kaldi/kaldi/src/decoder/kaldi-decoder.a        third_party/kaldi/kaldi/src/lm/kaldi-lm.a            third_party/kaldi/kaldi/src/transform/kaldi-transform.a
# third_party/kaldi/kaldi/src/feat/kaldi-feat.a              third_party/kaldi/kaldi/src/matrix/kaldi-matrix.a    third_party/kaldi/kaldi/src/tree/kaldi-tree.a
# third_party/kaldi/kaldi/src/fstext/kaldi-fstext.a          third_party/kaldi/kaldi/src/nnet2/kaldi-nnet2.a      third_party/kaldi/kaldi/src/util/kaldi-util.a
# third_party/kaldi/kaldi/src/gmm/kaldi-gmm.a                third_party/kaldi/kaldi/src/nnet3/kaldi-nnet3.a
# third_party/kaldi/kaldi/src/hmm/kaldi-hmm.a                third_party/kaldi/kaldi/src/nnet/kaldi-nnet.a


cc_import(name = "kaldi_lib",
          static_library = "libtensorflow_ext_op_lib.a",
          alwayslink=1)


# cc_library(name = "tensorflow_ext_op_lib",
#            srcs = ["kaldi_dataset_ops.cc.o"],
#            # static_library = "libtensorflow_ext_op_lib.a",
#            alwayslink=1)


tf_gen_op_wrapper_py("kaldi_table_dataset",
                     out="gen_kaldi_table_dataset.py",
                     deps=[":tensorflow_ext_op_lib"])