#!/usr/bin/env python3

import os
import re
import subprocess

from absl import app
from absl import flags

import tensorflow as tf

from galvASR.python import kaldi_environment
from galvASR.python.tensorflow_ext import kaldi_table_dataset, tdnn

FLAGS = flags.FLAGS

flags.DEFINE_integer('stage', 0, None)
flags.DEFINE_string('lang_dir', 'data/lang_test_tgsmall', None)
flags.DEFINE_string('train_data_dir', 'data/train_clean_5_sp_hires', None)
flags.DEFINE_string('train_ali_dir', 'exp/tri3b_ali_train_clean_5_sp', None)
flags.DEFINE_string('eval_data_dir', 'data/dev_clean_2', None)
flags.DEFINE_string('eval_ali_dir', 'exp/tri3b_ali_dev_clean_2', None)
flags.DEFINE_string('validate_data_dir', 'data/dev_clean_2', None)
flags.DEFINE_string('work_dir', 'exp/wav2letter', None)

flags.DEFINE_integer('batch_size', 16, None)
flags.DEFINE_integer('num_repeats', 1, None)

flags.DEFINE_float('learning_rate', 0.001700, None)


def create_label_indices(num_examples_tensor, eg_size, left_context):
  assert num_examples_tensor.shape.ndims == 0
  row_stride = tf.cumsum(
      tf.ones(num_examples_tensor, tf.int32) * eg_size, exclusive=True)
  col_stride = tf.cumsum(tf.ones((eg_size,), tf.int32), exclusive=True)
  X, Y = tf.meshgrid(col_stride, row_stride)
  label_indices_shape = tf.stack((num_examples_tensor, tf.constant(eg_size)))
  ones = tf.ones(label_indices_shape, tf.int32)
  return ones * left_context + X + Y


def create_feature_indices(num_examples_tensor, eg_size, total_context):
  row_stride = tf.cumsum(
      tf.ones(num_examples_tensor, tf.int32) * eg_size, exclusive=True)
  col_stride = tf.cumsum(
      tf.ones((eg_size + total_context,), tf.int32), exclusive=True)
  X, Y = tf.meshgrid(col_stride, row_stride)
  return X + Y

def input_and_label_datasets(data_dir, ali_dir):
  input_dataset = kaldi_table_dataset.KaldiFloat32MatrixDataset(
    "scp:" + os.path.join(data_dir, 'feats.scp'))
  with open(f"{ali_dir}/num_jobs") as fh:
    num_jobs = int(fh.read())
  archives = " ".join(
    f"{ali_dir}/ali.{i}.gz" for i in range(1, num_jobs + 1))
  label_dataset = kaldi_table_dataset.KaldiInt32VectorDataset(
    f"ark:gunzip -c {archives} |")

  return input_dataset, label_dataset

def main(_):
  if FLAGS.stage <= 0:
    num_labels = int(
        re.search(
            r'num-pdfs ([0-9]+)',
            subprocess.check_output(
                ["tree-info",
                 f"{FLAGS.train_ali_dir}/tree"]).decode('utf-8')).group(1))

    eg_size = 25
    params = {
        'hidden_layer_dims': [512],
        'num_splices': [3, 3],
        'dilations': [2, 2],
        'num_labels': num_labels,
        'learning_rate': 1e-5
    }

    model = tf.estimator.Estimator(
        tdnn.conv1d_tdnn,
        model_dir=os.path.join(FLAGS.work_dir, 'model_dir'),
        params=params)

    train_input_dataset, train_label_dataset = input_and_label_datasets(
      FLAGS.train_data_dir, FLAGS.train_ali_dir)
    eval_input_dataset, eval_label_dataset = input_and_label_datasets(
      FLAGS.eval_data_dir, FLAGS.eval_ali_dir)
    train_spec = tf.estimator.TrainSpec(
      lambda: create_dataset(train_input_dataset, train_label_dataset, eg_size),
      hooks=[
        tf.train.ProfilerHook(
          save_steps=100,
          output_dir=os.path.join(FLAGS.work_dir, 'profile_dir'))
      ])
    eval_spec = tf.estimator.EvalSpec(
      lambda: create_dataset(eval_input_dataset, eval_label_dataset, eg_size)
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def create_dataset(input_dataset, label_dataset, eg_size):

  def mapper(features_tuple, labels_tuple):
    dilations = params['dilations']
    num_splices = params['num_splices']
    total_context = sum(
      [(num_splice - 1) * dilation
       for num_splice, dilation in zip(num_splices, dilations)])
    assert total_context % 2 == 0
    left_context = total_context // 2
    labels = labels_tuple[1]
    num_examples_tensor = (tf.shape(labels)[0] - total_context) // eg_size

    features = features_tuple[1]

    assert_op = tf.Assert(
      tf.equal(tf.shape(features)[0],
               tf.shape(labels)[0]), [
                 tf.shape(features)[0],
                 tf.shape(labels)[0], features_tuple[0], labels_tuple[0]
               ])
    with tf.control_dependencies([assert_op]):
      feature_indices = create_feature_indices(num_examples_tensor, eg_size,
                                               total_context)
      features_egs = tf.gather(
        features, feature_indices, validate_indices=True)
      features_egs.set_shape((None, None, 40))

      label_indices = create_label_indices(num_examples_tensor, eg_size,
                                           left_context)
      labels_egs = tf.gather(labels, label_indices, validate_indices=True)

      return (features_egs, labels_egs)

    dataset = (tf.data.Dataset.zip(
      (input_dataset, label_dataset)).map(mapper).flat_map(
        lambda x, y: tf.data.Dataset.from_tensor_slices((x, y))).batch(
          FLAGS.batch_size).prefetch(1).repeat(FLAGS.num_repeats))
    return dataset

if __name__ == '__main__':
  kaldi_environment.setup_environment()
  with kaldi_environment.load_utils_and_steps():
    app.run(main)
