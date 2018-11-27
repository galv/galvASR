#include <memory>
#include <stdexcept>
#include <sstream>

#include "caffe2/core/db.h"
#include "caffe2/proto/caffe2.pb.h"

#include "kaldi/src/util/kaldi-holder.h"
#include "kaldi/src/util/kaldi-table.h"

namespace galvASR {

// We don't need to template on the type of the contents of the kaldi
// archive because Caffe2 does not make it the cursor's responsibility
// to deserialize.
template<typename Holder>
class KaldMatrixiDBCursor final : public caffe2::db::Cursor {
 public:
  explicit KaldiMatrixDBCursor(std::string r_specifier): reader_(r_specifier) {}
  ~KaldiMatrixDBCursor() override { CHECK(reader_.Close()); }

  void Seek(const std::string& key) override {
    throw std::logic_error("Seek not supported.");
  }
  // Note: We could support seek() if we used a RandomAccessTableReader
  bool SupportsSeek() override { return false; }
  void SeekToFirst() override { CHECK(reader_.Close());
    throw std::logic_error("Figure out later");
    // CHECK(reader_.Open());
  }
  void Next() override { reader_.Next(); }
  std::string key() override { return reader_.Key(); }
  std::string value() override {
    std::stringstream str_stream;
    const typename Holder::T &value = reader_.Value();
    value.Resize(value.NumRows(), value.NumCols(), kaldi::kCopyData,
                 kaldi::kStrideEqualNumCols);
    // TODO: Infer true or false-ness of binary mode from the
    // rspecifier and wspecifier
    TensorSerializer serializer;
    caffe2::Tensor<CPUContext> tensor;
    tensor.Resize(value.NumRows(), value.NumCols());
    tensor.ShareExternalPointer(value.Data());
    caffe2::TensorProtos protos;
    protos.add_protos()->;
    protos.mutable_protos(0)
  }
  bool Valid() override { return ! reader_.Done(); }

 private:
  kaldi::SequentialTableReader<Holder> reader_;
  Holder holder_;
};

template<typename Holder>
class KaldiMatrixDBTransaction final : public caffe2::db::Transaction {
 public:
  KaldiMatrixDBTransaction(const std::string& w_specifier)
    : writer_(w_specifier) { }
  ~KaldiMatrixDBTransaction() override { }
  /**
   * Puts the key value pair to the database.
   */
  void Put(const std::string& key, const std::string& value) override {
    std::stringstream value_stream(value);
    CHECK(holder_.Read(value_stream));
    writer_.Write(key, holder_.Value());
  }
  /**
   * Commits the current writes.
   */
  void Commit() override {
    writer_.Flush();
  }

 private:
  kaldi::TableWriter<Holder> writer_;
  Holder holder_;
};

template<typename Holder>
class KaldiMatrixDB final : public caffe2::db::DB {
 public:
  KaldiMatrixDB(const std::string& ark_specifier, caffe2::db::Mode mode);
  ~KaldiMatrixDB() override;

  void Close() override;
  std::unique_ptr<caffe2::db::Cursor> NewCursor() override;
  std::unique_ptr<caffe2::db::Transaction> NewTransaction() override;
 private:
  std::string ark_specifier_;
};

template<typename Holder>
KaldiMatrixDB<Holder>::KaldiMatrixDB(const std::string& ark_specifier, caffe2::db::Mode mode)
  : caffe2::db::DB(ark_specifier, mode),
    ark_specifier_(ark_specifier) { }

template<typename Holder>
KaldiMatrixDB<Holder>::~KaldiMatrixDB() { }

template<typename Holder>
void KaldiMatrixDB<Holder>::Close() { }

template<typename Holder>
std::unique_ptr<caffe2::db::Cursor> KaldiMatrixDB<Holder>::NewCursor() {
  if(mode_ != caffe2::db::READ) {
    auto message = "READ mode required to create a cursor.";
    throw std::invalid_argument(message);
  }
  std::unique_ptr<caffe2::db::Cursor> ptr(new KaldiMatrixDBCursor<Holder>(ark_specifier_));
  return ptr;
}

template<typename Holder>
std::unique_ptr<caffe2::db::Transaction> KaldiMatrixDB<Holder>::NewTransaction() {
  // TODO: Can you write to an existing Kaldi table?
  if(mode_ != caffe2::db::NEW) {
    auto message = "NEW mode required to create a transaction.";
    throw std::invalid_argument(message);
  }
  std::unique_ptr<caffe2::db::Transaction>
    ptr(new KaldiMatrixDBTransaction<Holder>(ark_specifier_));
  return ptr;
}





} // namespace galvASR
