import numpy as np


def serialize_kaldi_text(array):
  assert isinstance(array, np.ndarray)
  assert array.ndim == 1
  # return str(array).replace("[", "[ ").replace("]", " ]")
  return str(array).replace("[", "").replace("]", "")
