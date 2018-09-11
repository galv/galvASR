#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "kaldi/src/util/kaldi-holder.h"
#include "kaldi/src/util/kaldi-table.h"

#include <array>

namespace galvASR { namespace tensorflow_ext {

using namespace tensorflow;

// KALDI_ASSERT, KALDI_ERROR, and so on throw std::runtime_error.
using kaldi_error = std::runtime_error;

enum class KaldiType;

template<class Holder>
struct TFData;

Tensor getValueAsTensor(void *value, KaldiType type);

// Do I really need to implement the Dataset API? Seems like I could
// make a dataset of paths to my kaldi tables, and then map them
// through a regular parser operation... right?
template<class Holder>
class KaldiTableDatasetOp : public DatasetOpKernel {
 public:
  explicit KaldiTableDatasetOp(OpKernelConstruction* ctx) : DatasetOpKernel(ctx) { }

  void MakeDataset(OpKernelContext* ctx, DatasetBase** output) override {
    const Tensor* r_specifier_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("r_specifier", &r_specifier_tensor));
    OP_REQUIRES(ctx, r_specifier_tensor->dims() == 0 &&
                r_specifier_tensor->NumElements() == 1,
                errors::InvalidArgument("May not specify more than one r_specifier"));
    std::string r_specifier(r_specifier_tensor->flat<std::string>()(0));
    *output = new Dataset(ctx, r_specifier);
  }

 private:
  class Dataset : public GraphDatasetBase {
   public:
    Dataset(OpKernelContext* ctx, const string& r_specifier)
      : GraphDatasetBase(ctx), r_specifier_(r_specifier) { }

    std::unique_ptr<IteratorBase> MakeIteratorInternal(
        const string& prefix) const override {
      return std::unique_ptr<IteratorBase>(
          new Iterator({this, strings::StrCat(prefix, "::KaldiTable")}));
    }

    const DataTypeVector& output_dtypes() const override {
      static DataTypeVector* dtypes = new DataTypeVector({TFData<Holder>::dt});
      return *dtypes;
    }

    const std::vector<PartialTensorShape>& output_shapes() const override {
      // For some reason, we need to make a static local variable to
      // hold ::shape to force the compiler to make a symbol that it
      // can link to, even though this compile-time constant could be inlined.
      static auto shape_array = TFData<Holder>::shape;
      static std::vector<PartialTensorShape>* shapes = new std::vector<PartialTensorShape>({PartialTensorShape(gtl::ArraySlice<tensorflow::int64>(shape_array))});
      return *shapes;
    }

    string DebugString() const override { return "KaldiTableDatasetOp::Dataset"; }

   private:
    std::string r_specifier_;

   private:
    class Iterator : public DatasetIterator<Dataset> {
      // There must be a better way to be able to access the Params type...
      typedef typename tensorflow::DatasetIterator<KaldiTableDatasetOp<Holder>::Dataset>::Params
      MyParams;

     public:
      // Do I need to do something here?
      explicit Iterator(const MyParams& params)
        : DatasetIterator<Dataset>(params) { }
      // When does the destructor get called anyway?
      ~Iterator() override {
        mutex_lock l(mu_);
        if (reader_initialized_) {
          bool failure_occurred = reader_.Close();
          if (failure_occurred) {
            LOG(ERROR) << this->dataset()->r_specifier_ <<
              " done early because of an error";
          }
        }
      }

      Status GetNextInternal(IteratorContext* /*ctx*/,
                             std::vector<Tensor>* out_tensors,
                             bool* end_of_sequence) override {
        mutex_lock l(mu_);
        *end_of_sequence = false;
        if (!reader_initialized_) {
          if (!reader_.Open(this->dataset()->r_specifier_)) {
            std::stringstream sstr;
            sstr << "Failed to open: " << this->dataset()->r_specifier_;
            LOG(ERROR) << sstr.str();
            return Status(error::NOT_FOUND, sstr.str());
          }
          VLOG(10) << "Succesfully opened: " << this->dataset()->r_specifier_;
          reader_initialized_ = true;
        }
        if (reader_.Done()) { *end_of_sequence = true; return Status::OK(); }

        try {
          out_tensors->emplace_back(std::move(getValueAsTensor(&reader_.Value(), TFData<Holder>::type)));
        } catch (const kaldi_error& exception) {
          std::stringstream sstr;
          sstr << "Failed to read " << reader_.Key() << " in: " <<
            this->dataset()->r_specifier_;
          return Status(error::NOT_FOUND, sstr.str());
        }
        reader_.Next();
        return Status::OK();
      }


     private:
      mutex mu_;
      kaldi::SequentialTableReader<Holder> reader_ GUARDED_BY(mu_);
      bool reader_initialized_ GUARDED_BY(mu_) = false;
    };
  };
};

enum class KaldiType {
  FloatMatrix,
  FloatVector,
  Int32Vector
};

template<class Holder>
struct TFData {
};

template<>
struct TFData<kaldi::KaldiObjectHolder<kaldi::Matrix<kaldi::float32>>> {
  static constexpr KaldiType type = KaldiType::FloatMatrix;
  static constexpr DataType dt = DT_FLOAT;
  static constexpr std::array<tensorflow::int64, 2> shape{{-1, -1}};
};

template<>
struct TFData<kaldi::KaldiObjectHolder<kaldi::Vector<kaldi::float32>>> {
  static constexpr KaldiType type = KaldiType::FloatVector;
  static constexpr DataType dt = DT_FLOAT;
  static constexpr std::array<tensorflow::int64, 1> shape{{-1}};
};

template<>
struct TFData<kaldi::BasicVectorHolder<kaldi::int32>> {
  static constexpr KaldiType type = KaldiType::Int32Vector;
  static constexpr DataType dt = DT_INT32;
  static constexpr std::array<tensorflow::int64, 1> shape{{-1}};
};

Tensor getValueAsTensor(void *value, KaldiType type) {
  switch (type) {
    case KaldiType::FloatMatrix: {
      kaldi::Matrix<float>& matrix = *static_cast<kaldi::Matrix<float>*>(value);
      matrix.Resize(matrix.NumRows(), matrix.NumCols(), kaldi::kCopyData,
                    kaldi::kStrideEqualNumCols);
      Tensor tensor(DT_FLOAT, TensorShape({matrix.NumRows(), matrix.NumCols()}));
      auto flat = tensor.flat<float>();
      std::copy_n(matrix.Data(), matrix.NumRows() * matrix.NumCols(),
                  flat.data());
      return tensor;
    }
    case KaldiType::FloatVector: {
      kaldi::Vector<float>& vector = *static_cast<kaldi::Vector<float>*>(value);
      Tensor tensor(DT_FLOAT, TensorShape({vector.Dim()}));
      auto flat = tensor.flat<float>();
      std::copy_n(vector.Data(), vector.Dim(), flat.data());
      return tensor;
    }
    case KaldiType::Int32Vector: {
      std::vector<kaldi::int32>& vector =
        *static_cast<std::vector<kaldi::int32>*>(value);
      // Use tensorflow:: prefix because kaldi and openfst also define
      // their own int64's.
      tensorflow::int64 size = vector.size();
      Tensor tensor(DT_INT32, TensorShape({size}));
      auto flat = tensor.flat<kaldi::int32>();
      std::copy_n(vector.data(), vector.size(), flat.data());
      return tensor;
    }
  }
}

// Use a macro to register ops for the different kinds of tables.

// Note: I struggle to use the correct do/while(0) construct for these
// macros for whatever reason. I get: "expected unqualified-id" from gcc.
// This is because you can't use do {} while(0) in global scope.
// Would have to use a constructor instead.
#define REGISTER_DATASET_KERNEL(op_name, holder_type)            \
  REGISTER_KERNEL_BUILDER(Name(op_name)                          \
                          .Device(DEVICE_CPU),                   \
                          KaldiTableDatasetOp<holder_type>)
REGISTER_DATASET_KERNEL("KaldiFloat32MatrixDataset",
                                 kaldi::KaldiObjectHolder<kaldi::Matrix<kaldi::float32>>)
REGISTER_DATASET_KERNEL("KaldiFloat32VectorDataset",
                                 kaldi::KaldiObjectHolder<kaldi::Vector<kaldi::float32>>)
REGISTER_DATASET_KERNEL("KaldiInt32VectorDataset",
                                 kaldi::BasicVectorHolder<kaldi::int32>)

#undef REGISTER_DATASET_KERNEL

} // namespace tensorflow_ext
} // namespace galvASR
