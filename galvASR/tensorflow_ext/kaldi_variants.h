#include "kaldi/src/lat/kaldi-lattice.h"

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"

namespace galvASR {

class LatticeVariant {
public:
  // DefaultConstructible
  LatticeVariant() = default;
  ~LatticeVariant();
  // CopyConstructible
  LatticeVariant(const LatticeVariant&);
  LatticeVariant(kaldi::Lattice *lattice_ptr);
  std::string TypeName() const;
  std::string DebugString() const;
  void Encode(tensorflow::VariantTensorData* data) const;
  bool Decode(tensorflow::VariantTensorData data);
private:
  kaldi::Lattice *lattice_ptr_;
};

static_assert(std::is_copy_constructible<LatticeVariant>::value, "Bad!");

LatticeVariant::~LatticeVariant() {
  delete lattice_ptr_;
}

LatticeVariant::LatticeVariant(kaldi::Lattice *lattice_ptr):
  lattice_ptr_(lattice_ptr) { }

LatticeVariant::LatticeVariant(const LatticeVariant& other) {
  lattice_ptr_ = new kaldi::Lattice(*other.lattice_ptr_);
}

std::string LatticeVariant::TypeName() const {
  return "galvASR::LatticeVariant";
}

std::string LatticeVariant::DebugString() const {
  return "My Debug String";
}

void LatticeVariant::Encode(tensorflow::VariantTensorData* data) const {
  std::stringstream serialization_stream;
  CHECK(kaldi::WriteLattice(serialization_stream, true, *lattice_ptr_));
  data->type_name_ = TypeName();
  data->metadata_ = serialization_stream.str();
}

// TODO(galv): Why is this using pass-by-value? Seems expensive...
// Unless template magic inlines it and thereby elides the copy...
// Actually variant_encode_decode.h uses std::move on its argument... Strange!
bool LatticeVariant::Decode(tensorflow::VariantTensorData data) {
  std::stringstream input_stream(data.metadata_);
  if (data.type_name_ != TypeName()) {
    return false;
  }
  kaldi::Lattice *lattice_ptr = nullptr;
  if (!kaldi::ReadLattice(input_stream, true, &lattice_ptr)) {
    return false;
  }
  lattice_ptr_ = lattice_ptr;
  return true;
}

} // namespace galvASR
