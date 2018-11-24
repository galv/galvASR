import os

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape

_op_lib = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          'libtensorflow_ext.so'))


class KaldiFloat32MatrixDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiFloat32MatrixDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return _op_lib.kaldi_float32_matrix_dataset(self._r_specifier)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return tf.Tensor, tf.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([]),
            tensor_shape.TensorShape([None, None]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.float32


class KaldiFloat32VectorDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiFloat32VectorDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return _op_lib.kaldi_float32_vector_dataset(self._r_specifier)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return tf.Tensor, tf.Tensor

  @property
  def output_shapes(self):
    return  (tensor_shape.TensorShape([]),
             tensor_shape.TensorShape([None]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.float32


class KaldiInt32VectorDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiInt32VectorDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return _op_lib.kaldi_int32_vector_dataset(self._r_specifier)

  def _inputs(self):
    return []

  @property
  def output_classes(self):
    return tf.Tensor, tf.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([]),
            tensor_shape.TensorShape([None]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.int32


class KaldiWaveDataset(Dataset):
  def __init__(self, r_specifier):
    super(KaldiWaveDataset, self).__init__()
    self._r_specifier = r_specifier

  def _inputs(self):
    return []

  def _as_variant_tensor(self):
    return _op_lib.kaldi_wave_dataset(self._r_specifier)

  @property
  def output_classes(self):
    return tf.Tensor, tf.Tensor

  @property
  def output_shapes(self):
    return (tensor_shape.TensorShape([]),
            tensor_shape.TensorShape([None, None]))

  @property
  def output_types(self):
    return dtypes.string, dtypes.float32


def key_func(data):
  string_key = data[0]
  return tf.strings.to_hash_bucket_fast(string_key)

def reduce_func():
  pass


WINDOW_SIZE = 2

def reduce_func(key, same_key_dataset):
  features = same_key_dataset[0][1]
  label = same_key_dataset[0][2]
  assert key.dtype == tf.int64
