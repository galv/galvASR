import tensorflow as tf

import os


class KaldiIOTest(tf.test.TestCase):
    def testKaldiIO(self):
        kaldi_io = tf.load_op_library(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../libtensorflow_ext.so'))
        print(dir(kaldi_io))
        with self.test_session():
            dataset = kaldi_io.kaldi_table_dataset('feats.scp')
            print(type(dataset))
            print(dataset)
            print(dataset.output_types)
            print(dataset.output_shapes)
            pass


if __name__ == "__main__":
    tf.test.main()
