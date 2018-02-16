#include <memory>
#include <stdexcept>
#include <sstream>

#include "caffe2/core/db.h"
#include "caffe2/proto/caffe2.pb.h"

#include "kaldi/src/util/kaldi-holder.h"
#include "kaldi/src/util/kaldi-table.h"

namespace galvASR {

namespace {

template<typename Matrix>
struct VectorToTypeMeta {
  
};

template<>
struct VectorToTypeMeta<kaldi::Vector<float>> {
  static constexpr id = TypeMeta::Id<float>();
};

template<>
struct VectorToTypeMeta<kaldi::Vector<double>> {
  static constexpr id = TypeMeta::Id<double>();
};

}

template<typename Holder>
class KaldiVectoriDBCursor final : public caffe2::db::Cursor {
 public:
  explicit KaldiVectorDBCursor(std::string r_specifier): reader_(r_specifier) {}
  ~KaldiVectorDBCursor() override { CHECK(reader_.Close()); }

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
    // TODO: Infer true or false-ness of binary mode from the
    // rspecifier and wspecifier
    // caffe2::Tensor<CPUContext> tensor;
    // tensor.Resize(value.Dim());
    // tensor.ShareExternalPointer(value.Data());
    caffe2::TensorProtos protos;
    caffe2::TensorProto *proto = protos.add_protos();
    proto->set_name(this->key());
    proto->set_data_type(TypeMetaToDataType(MatrixToTypeMeta<Holder::T>));
    proto->add_dims(value.Dim());
    proto->mutable_float_data()->Reserve(value.Dim());
    memcpy(proto->mutable_float_data().mutable_data(), value.Data(),
           value.Dim());
    return protos.SerializeAsString(protos);
  }
  bool Valid() override { return ! reader_.Done(); }

 private:
  kaldi::SequentialTableReader<Holder> reader_;
  Holder holder_;
};

template<typename Holder>
class KaldiVectorDBTransaction final : public caffe2::db::Transaction {
 public:
  KaldiVectorDBTransaction(const std::string& w_specifier)
    : writer_(w_specifier) { }
  ~KaldiVectorDBTransaction() override { }
  /**
   * Puts the key value pair to the database.
   */
  void Put(const std::string& key, const std::string& value) override {
    TensorProtos protos;
    TensorDeserializer<CPUContext> deserializer;
    deserializer.Deserialize();
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
  TensorDeserializer<CPUContext> deserializer;
};

template<typename Holder>
class KaldiVectorDB final : public caffe2::db::DB {
 public:
  KaldiVectorDB(const std::string& ark_specifier, caffe2::db::Mode mode);
  ~KaldiVectorDB() override;

  void Close() override;
  std::unique_ptr<caffe2::db::Cursor> NewCursor() override;
  std::unique_ptr<caffe2::db::Transaction> NewTransaction() override;
 private:
  std::string ark_specifier_;
};

template<typename Holder>
KaldiVectorDB<Holder>::KaldiVectorDB(const std::string& ark_specifier, caffe2::db::Mode mode)
  : caffe2::db::DB(ark_specifier, mode),
    ark_specifier_(ark_specifier) { }

template<typename Holder>
KaldiVectorDB<Holder>::~KaldiVectorDB() { }

template<typename Holder>
void KaldiVectorDB<Holder>::Close() { }

template<typename Holder>
std::unique_ptr<caffe2::db::Cursor> KaldiVectorDB<Holder>::NewCursor() {
  if(mode_ != caffe2::db::READ) {
    auto message = "READ mode required to create a cursor.";
    throw std::invalid_argument(message);
  }
  std::unique_ptr<caffe2::db::Cursor> ptr(new KaldiVectorDBCursor<Holder>(ark_specifier_));
  return ptr;
}

template<typename Holder>
std::unique_ptr<caffe2::db::Transaction> KaldiVectorDB<Holder>::NewTransaction() {
  // TODO: Can you write to an existing Kaldi table?
  if(mode_ != caffe2::db::NEW) {
    auto message = "NEW mode required to create a transaction.";
    throw std::invalid_argument(message);
  }
  std::unique_ptr<caffe2::db::Transaction>
    ptr(new KaldiVectorDBTransaction<Holder>(ark_specifier_));
  return ptr;
}

} // namespace galvASR
