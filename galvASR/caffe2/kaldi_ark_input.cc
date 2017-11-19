namespace galvASR {

// Do we need to be in the caffe2 namespace to use this? That would be troublesome!
REGISTER_CPU_OPERATOR(KaldiFloatMatrixArchiveInput,
                      KaldiArkInputOp<kaldi::Matrix<float>, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiFloatVectorArchiveInput,
                      KaldiArkInputOp<kaldi::Vector<float>, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiDoubleMatrixArchiveInput,
                      KaldiArkInputOp<kaldi::Matrix<double>, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiDoubleVectorArchiveInput,
                      KaldiArkInputOp<kaldi::Vector<double>, CPUContext>);

// Special case! wav files, woohoo!
REGISTER_CPU_OPERATOR(KaldiDoubleVectorArchiveInput,
                      KaldiArkInputOp<kaldi::Vector<double>, CPUContext>);

// TODO: Replace this. A lot.
SHOULD_NOT_DO_GRADIENT(KaldiFloatMatrixArchiveInput);

// Documentation will be nearly the same among all of these. Ugh! So ugly!
OPERATOR_SCHEMA(KaldiFloatMatrixArchiveInput);
OPERATOR_SCHEMA(KaldiFloatVectorArchiveInput);
OPERATOR_SCHEMA(KaldiDoubleMatrixArchiveInput);
OPERATOR_SCHEMA(KaldiDoubleVectorArchiveInput);

// basic types
REGISTER_CPU_OPERATOR(KaldiFloatArchiveInput,
                      KaldiArkInputOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiDoubleArchiveInput,
                      KaldiArkInputOp<double, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiBoolArchiveInput,
                      KaldiArkInputOp<bool, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiIntArchiveInput,
                      KaldiArkInputOp<int, CPUContext>);

static_assert(sizeof(int) == 4, "Caffe2 assumes ints are 32-bit elsewhere. "
              "This must hold true in order to be compatible with Kaldi's "
              "architecture-independent int32 type.");

// continers of basic types
REGISTER_CPU_OPERATOR(KaldiIntVectorArchiveInput,
                      KaldiArkInputOp<std::vector<int>, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiIntVectorVectorArchiveInput,
                      KaldiArkInputOp<std::vector<std::vector<int>>, CPUContext>);
REGISTER_CPU_OPERATOR(KaldiIntPairVectorArchiveInput,
                      KaldiArkInputOp<std::vector<std::pair<int>>, CPUContext>);

}
