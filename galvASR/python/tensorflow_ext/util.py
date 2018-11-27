"""
Handy utility functions specific to Tensorflow
"""


def dimension_equal_if_defined(a, b):
  """
Args:
  a: tf.Dimension
  b: tf.Dimension

Returns:
  False if a and b are guaranteed not to be the same dimension. Otherwise True.
  """
  if a == tf.Dimension(None) or b == tf.Dimension(None):
    # Not enough information to decide. Return True
    return True
  else:
    return a == b


def is_likely_iterable(value):
  try:
    # TODO: Does this call some how change the state of value?
    iter(value)
  except TypeError:
    return False
  else:
    return True
