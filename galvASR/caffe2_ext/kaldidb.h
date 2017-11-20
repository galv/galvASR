namespace galvASR {

// We don't need to template on the type of the contents of the kaldi
// archive because Caffe2 does not make it the cursor's responsibility
// to deserialize.
class KaldiDBCursor : public caffe2::db::Cursor {
 public:
  explicit KaldiDBCursor(kaldi::) ;
  ~KaldiDBCursor() override;

  void Seek() override;
  bool SupportsSeek() override;
  void SeekToFirst() override;
  void Next() override;
  std::string key() override;
  std::string value() override;
  bool Valid() override;

 private:
  kaldi::SequentialTableReader
}
}
  
REGISTER_CAFFE2_DB(KaldiDB, KaldiDB);

}
