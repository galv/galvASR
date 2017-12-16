#include "kaldidb.h"

#include "caffe2/core/db.h"
#include "caffe2/utils/proto_utils.h"
#include "caffe2/core/logging.h"

#include "kaldi/src/feat/wave-reader.h"
// HACK: openfst defines its own implementations of glog and gflags,
// which conflict with the oneswe pull in from Caffe2. Ignore
// openfst's implementations by tricking C++ into thinking it's
// already included those files.
//#define FST_LIB_LOG_H_
//#define FST_LIB_FLAGS_H_
//#include "kaldi/src/fstext/kaldi-fst-io.h"
// #include "kaldi/src/chain/chain-supervision.h"

namespace galvASR {

namespace {
// These are required to get the below macro expansions to compile happily.
using namespace caffe2;
using namespace caffe2::db;

REGISTER_CAFFE2_DB(KaldiFloatMatrixDB,
                   KaldiDB<kaldi::KaldiObjectHolder<kaldi::Matrix<float>>>);
REGISTER_CAFFE2_DB(KaldiFloatVectorDB,
                   KaldiDB<kaldi::KaldiObjectHolder<kaldi::Vector<float>>>);

REGISTER_CAFFE2_DB(KaldiWaveDB,
                   KaldiDB<kaldi::WaveHolder>);

REGISTER_CAFFE2_DB(KaldiInt32DB,
                   KaldiDB<kaldi::BasicHolder<int32>>);

// REGISTER_CAFFE2_DB(KaldiVectorFstDB,
//                    KaldiDB<fst::VectorFstHolder>);

// REGISTER_CAFFE2_DB(KaldiChainSupervisionDB,
//                    KaldiDB<kaldi::KaldiObjectHolder<kaldi::chain::Supervision>>);
} // namespace

} // namespace galvASR
