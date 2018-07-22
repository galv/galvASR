#!/usr/bin/env python3

import os
import subprocess
import tempfile

from absl import app
from absl import flags

import tensorflow as tf

from galvASR.python import kaldi_environment

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, None)
flags.DEFINE_string('lang_dir', 'data/lang_test_tgsmall', None)
flags.DEFINE_string('ali_dir', 'exp/tri3_ali_train_clean_5_sp', None)
flags.DEFINE_string('train_data_dir', 'data/train_clean_5', None)
flags.DEFINE_string('validate_data_dir', 'data/dev_clean_2', None)
flags.DEFINE_string('work_dir', 'exp/wav2letter', None)

flags.DEFINE_integer('batch_size', 16, None)
flags.DEFINE_integer('num_repeats', 1, None)


def main(argv):
  if FLAGS.stage <= 0:
    with tempfile.NamedTemporaryFile('w+') as labels_fh:
      subprocess.check_call(["ali-to-pdf", os.path.join(FLAGS.ali_dir, 'final.mdl'),
                             "'ark:gunzip -c < {}/ali.1.gz|'".format(os.path.join(FLAGS.ali_dir)),
                             "ark,t:{}".format(labels_fh.name)])
      # ali-to-pdf ../../../../../../src/galvASR/examples/mini_librispeech/exp/tri3_ali_train_clean_5_sp/final.mdl "ark:gunzip -c < ../../../../../../src/galvASR/examples/mini_librispeech/exp/tri3_ali_train_clean_5_sp/ali.1.gz |" ark,t:-
      pdf_id_dataset = (tf.data.TextLineDataset(labels_fh.name)
                        .map(lambda line: tf.strings.split(line, sep=" "))
                        .)

    input_dataset = tf.data.TextLineDataset(os.path.join(FLAGS.lang_dir, "text"))
    input_dataset = kaldi_table_dataset.KaldiFloat32MatrixDataset(
      "scp:" + os.path.join(FLAGS.train_data_dir, 'feats.scp'))
    label_dataset = kaldi_table_dataset.KaldiInt32VectorDataset(
      " ".join("ark:utils/sym2int.pl", "--map-oov", oov, "-f", "2-", "<",
               words_txt, os.path.join(FLAGS.train_data_dir, 'text'), "|"))
    dataset = (tf.data.Dataset.zip((input_dataset, label_dataset))
               .batch(FLAGS.batch_size)
               .repeat(FLAGS.num_repeats)
               )

    tf.estimator.Estimator(tdnn_model_fn_with_cross_entropy)


if __name__ == '__main__':
  kaldi_environment.setup_environment()
  with kaldi_environment.load_utils_and_steps():
    app.run(main)
