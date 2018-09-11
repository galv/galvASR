import os

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

_op_lib = tf.load_op_library(
  os.path.join(os.path.dirname(os.path.realpath(__file__)),
               'libtensorflow_ext.so'))


class KaldiFloat32MatrixDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiFloat32MatrixDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return _op_lib.kaldi_float32_matrix_dataset(self._r_specifier)

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
    return _op_lib.kaldi_float32_vector_dataset(self._r_specifier)

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
    return _op_lib.kaldi_int32_vector_dataset(self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([None])

  @property
  def output_types(self):
    return dtypes.int32


class KaldiWaveDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiWaveDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return _op_lib.kaldi_wave_dataset(self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.TensorShape([None, None])

  @property
  def output_types(self):
    return dtypes.float32
