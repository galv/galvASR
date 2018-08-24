load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_gen_op_libs")

cc_import(name = "tensorflow_ext_op_lib",
          static_library = repository_ctx.os.environ.get("OP_LIB") + "/libtensorflow_ext_op_lib.a",
          # static_library = "libtensorflow_ext_op_lib.a",
          alwayslink=1)

tf_gen_op_wrapper_py("kaldi_table_dataset",
                     out="gen_kaldi_table_dataset.py",
                     deps=[":tensorflow_ext_op_lib", "@kaldi//:kaldi_lib",],
                     # This is hacky. We are assuming that kaldi is
                     # depending on ATLAS. An easy way to clean up is
                     # to make "@kaldi//:kaldi_lib" point to a shared
                     # object of all of the static archives
                     cc_linkopts=["-Bstatic", "-llapack_atlas", "-lcblas",
                     "-latlas", "-lf77blas"])