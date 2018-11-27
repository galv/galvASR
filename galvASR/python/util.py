"""
Handy utility functions specific to Tensorflow
"""


def is_likely_iterable(value):
  try:
    # TODO: Does this call some how change the state of value?
    iter(value)
  except TypeError:
    return False
  else:
    return True
