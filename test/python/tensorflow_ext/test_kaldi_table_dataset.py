import collections
import itertools
import tempfile

import tensorflow as tf
import numpy as np

from galvASR.python.tensorflow_ext import kaldi_table_dataset

def test_read():
  num_repeats = 2
  arrays = collections.OrderedDict(
      [("one", np.array([0, 1, 2], dtype=np.int32)),
       ("two", np.array([3, 4, 5], dtype=np.int32))])
  with tempfile.NamedTemporaryFile() as fh:
    for k, v in arrays.items():
      line = bytes("{} {}\n".format(k, serialize_kaldi_text(v)), 'utf-8')
      fh.write(line)
    fh.flush()
    dataset = (kaldi_table_dataset.KaldiInt32VectorDataset('ark,t:' + fh.name).
               repeat(num_repeats))
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as session:
      i = 0
      for ground_truth in itertools.chain.from_iterable(
          itertools.repeat(arrays.values(), num_repeats)):
        assert np.array_equal(ground_truth, session.run(next_element)[1])
        i += 1


def test_merge():
  x = collections.OrderedDict([("one", np.array([0, 1, 2], dtype=np.float32)),
                               ("two", np.array([3, 4, 5], dtype=np.float32)),
                               ("three", np.array([6, 7, 8],
                                                  dtype=np.float32))])
  y = collections.OrderedDict([("one", np.array([0, 1, 2], dtype=np.int32)),
                               ("three", np.array([3, 4, 5], dtype=np.int32))])
  with tempfile.NamedTemporaryFile() as fh_x, \
       tempfile.NamedTemporaryFile() as fh_y:
    for (fh, data) in ((fh_x, x), (fh_y, y)):
      for k, v in data.items():
        line = bytes("{} {}\n".format(k, serialize_kaldi_text(v)), 'utf-8')
        fh.write(line)
        fh.flush()
    x_data = kaldi_table_dataset.KaldiFloat32VectorDataset('ark,t:' + fh.name)
    y_data = kaldi_table_dataset.KaldiInt32VectorDataset('ark,t:' + fh.name)

    zipped_dataset = tf.data.Dataset.zip((x_data, y_data))
    print(zipped_dataset.output_shapes)
    print(zipped_dataset.output_types)


# Helpers

def serialize_kaldi_text(array):
  assert isinstance(array, np.ndarray)
  assert array.ndim == 1
  # return str(array).replace("[", "[ ").replace("]", " ]")
  return str(array).replace("[", "").replace("]", "")



