#!/usr/bin/env python3

import os
import subprocess

from absl import app
from absl import flags

import pywrapfst as openfst

import tensorflow as tf

from galvASR.python import kaldi_environment
from galvASR.python.fst_ext import spelling_fst
from galvASR.python.tensorflow_ext import kaldi_table_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, None)
flags.DEFINE_string('lang_dir', 'data/lang_test_tgsmall', None)
flags.DEFINE_string('train_data_dir', 'data/train_clean_5', None)
flags.DEFINE_string('validate_data_dir', 'data/dev_clean_2', None)
flags.DEFINE_string('work_dir', 'exp/wav2letter', None)
flags.DEFINE_string('repeat_letter', '2', None)
flags.DEFINE_integer('self_transition_prob', 0.5, None, lower_bound=0.0, upper_bound=1.0)

flags.DEFINE_integer('batch_size', 16, None)
flags.DEFINE_integer('num_repeats', 1, None)


def main(argv):
  training_input_dir = os.path.join(FLAGS.work_dir, 'training_inputs')
  os.makedirs(training_input_dir, exist_ok=True)
  words_txt = os.path.join(FLAGS.lang_dir, 'words.txt')
  if FLAGS.stage <= 0:
    word_table = openfst.SymbolTable.read_text(words_txt)
    alphabet_table = spelling_fst.create_alphabet_symbol_table(word_table)
    S = spelling_fst.create_spelling_fst(word_table, alphabet_table,
                                         FLAGS.repeat_letter)
    G = openfst.Fst.read(os.path.join(FLAGS.lang_dir, 'G.fst'))
    SG = openfst.determinize(openfst.compose(S, G))
    SG.minimize()

    S.write(os.path.join(training_input_dir), 'S.fst')
    SG.write(os.path.join(training_input_dir), 'SG.fst')

  with open(os.path.join(FLAGS.lang_dir, 'oov.txt')) as oov_fh:
    oov = oov_fh.read().strip()
  if FLAGS.stage <= 1:
    results = subprocess.run(["utils/sym2int.pl", "--map-oov", oov, "-f", "2-",
                              "<", words_txt,
                              os.path.join(FLAGS.train_data_dir, 'text')],
                             stdout=subprocess.PIPE, check=True,
                             universal_newlines=True)
    openfst.FarWriter()
    for line in results.stdout:
      key, value = line.split(None, maxsplit=1)
      value = [int(index) for index in value.split()]
      value

  neural_net_dir = os.path.join(FLAGS.work_dir, 'nnet')
  os.makedirs(neural_net_dir, exist_ok=True)
  if FLAGS.stage <= 2:
    input_dataset = kaldi_table_dataset.KaldiFloat32MatrixDataset(
      "scp:" + os.path.join(FLAGS.train_data_dir, 'feats.scp'))
    label_dataset = kaldi_table_dataset.KaldiInt32VectorDataset(
      " ".join("ark:utils/sym2int.pl", "--map-oov", oov, "-f", "2-", "<",
               words_txt, os.path.join(FLAGS.train_data_dir, 'text'), "|"))
    dataset = (tf.data.Dataset.zip((input_dataset, label_dataset))
               .batch(FLAGS.batch_size)
               .repeat(FLAGS.num_repeats)
               )
    

if __name__ == '__main__':
  kaldi_environment.setup_environment()
  with kaldi_environment.load_utils_and_steps():
    app.run(main)
