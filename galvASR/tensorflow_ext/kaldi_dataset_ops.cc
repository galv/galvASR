namespace {

using namespace tensorflow;

template<typename Holder>
class KaldiTableDatasetOp : public DatasetOpKernel(ctx) {
 public:
  explicit KaldiDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_types", &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("output_shapes", &output_shapes_));
    for (const DataType& dt : output_types_) {
      OP_REQUIRES(ctx,
                  dt == DT_STRING || dt == DT_BOOL || 
                  dt == DT_INT32 || dt == DT_INT64 ||
                  dt == DT_FLOAT || dt == DT_DOUBLE,
                  errors::InvalidArgument(
                    "Each element of `output_types_` must be one of: "
                    "DT_STRING, DT_BOOL, DT_INT32, DT_INT64, "
                    "DT_FLOAT, or DT_DOUBLE"));
    }
  }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    *output = new Dataset()
  }

 private:
  class Dataset : public DatasetBase {
   public:
    Dataset(const string& r_specifier, const DataTypeVector& output_types,
            const std::vector<PartialTensorShape>& output_shapes)
      : r_specifier_(r_specifier)

    std::unique_ptr<IteratorBase> MakeIterator(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::KaldiTable")}));
    }

    const DataTypeVector& output_dtypes() const override {
      return output_types_;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      return output_shapes_;
    }

    string DebugString() override { return "KaldiTableDatasetOp::Dataset"; }  

   private:
    class Iterator : public DatasetIterator<Dataset> {
     public:
      explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params) {}
      ~Iterator() override {
        if (reader_initialized_) {
          bool failure_occurred = reader_.Close();
          LOG(ERROR) << dataset()->r_specifier_ <<
            " done early because of an error";
        }
      }

      Status GetNextInternal(IteratorContext* /*ctx*/,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        if (!reader_initialized_) {
          if (!reader_.Open(dataset()->r_specifier_)) {
            std::strinstream sstr;
            sstr << "Failed to open: " << dataset()->r_specifier_;
            LOG(ERROR) << sstr.str();
            return Status(error::NOT_FOUND, sstr.str());
          }
          reader_initialized_ = true;
        }

        try {
          out_tensors[0] = std::move(getValueAsTensor(&reader_.Value()));
        } catch (const std::runtime_exception& exception) {
          std::strinstream sstr;
          sstr << "Failed to read " << reader_.Key() << " in: " <<
            dataset()->r_specifier_;
          return Status(error::NOT_FOUND, sstr.str());
        }
        reader_.Next();
        if (reader_.Done()) { *end_of_sequence = true; }
        return Status::OK();
      }


     private:
      mutext mu_;
      kaldi::SequentialTableReader<Holder> reader_ GUARDED_BY(mu_);
      bool reader_initialized_ GUARDED_BY(mu_) = false;
    };
  };
};

enum class KaldiType {
  FloatMatrix;
  FloatVector;
  Int32Vector;
};

template<typename Holder>
struct HolderToEnum {

};

template<>
struct HolderToEnum<KaldiObjectHolder<kaldi::Matrix<float>>> {
  static constexpr KaldiType type = KaldiType::FloatMatrix;
};

template<>
struct HolderToEnum<KaldiObjectHolder<kaldi::Vector<float>>> {
  static constexpr KaldiType type = KaldiType::FloatVector;
};

template<>
struct HolderToEnum<KaldiObjectHolder<std::vector<int32>>> {
  static constexpr KaldiType type = KaldiType::Int32Vector;
};

Tensor getValueAsTensor(void *value, KaldiType type)
  switch type {
    case KaldiType::FloatMatrix:
      kaldi::Matrix<float>& matrix = *value;
      matrix.Resize(matrix.NumRows(), matrix.NumCols(), kaldi::kCopyData,
                    kaldi::kStrideEqualNumCols);
      Tensor tensor(DT_FLOAT, TensorShape({matrix.NumRows(), matrix.NumCols()}));
      Tensor flat = tensor->flat<float>();
      std::copy_n(matrix.Data(), matrix.NumRows() * matrix.NumCols(),
                  flat.data());
      return tensor;
    case KaldiType::FloatVector:
      kaldi::Vector<float>& vector = *value;
      Tensor tensor(DT_FLOAT, TensorShape({vector.Dim()}));
      std::copy_n(vector.Data(), vector.Dim(), tensor.data());
      return tensor;
    case KaldiType::Int32Vector:
      std::vector<int32>& vector = *value;
      Tensor tensor(DT_INT32, TensorShape({vector.size()}));
      std::copy_n(vector.data(), vector.size(), tensor.data());
      return tensor;
  }
}
