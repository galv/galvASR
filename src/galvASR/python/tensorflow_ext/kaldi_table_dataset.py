import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

from galvASR.python.tensorflow_ext._gen_dataset_ops import kaldi_table_dataset_with_op_name


class KaldiFloat32MatrixDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiFloat32MatrixDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return kaldi_table_dataset_with_op_name("KaldiFloat32MatrixDataset", self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([None, None])

  @property
  def output_types(self):
    return dtypes.float32


class KaldiFloat32VectorDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiFloat32VectorDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return kaldi_table_dataset_with_op_name("KaldiFloat32VectorDataset", self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([None])

  @property
  def output_types(self):
    return dtypes.float32


class KaldiInt32VectorDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiInt32VectorDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return kaldi_table_dataset_with_op_name("KaldiInt32VectorDataset", self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([None])

  @property
  def output_types(self):
    return dtypes.int32
