from tensorflow.python.data.ops.dataset_ops import Dataset

from galvASR.python.tensorflow._gen_dataset_ops import kaldi_table_dataset


kaldi_io = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../tensorflow_ext/libtensorflow_ext.so'))


class KaldiTableDataset(Dataset):

  def __init__(self, r_specifier):
    super(KaldiTableDataset, self).__init__()
    self._r_specifier = r_specifier

  def _as_variant_tensor(self):
    return kaldi_table_dataset(self._r_specifier)

  def output_shapes(self):
    # ? Take a look at readers.py in tensorflow
    raise NotImplementedError("KaldiTableDataset.output_shapes")

  def output_types(self):
    raise NotImplementedError("KaldiTableDataset.output_types")
