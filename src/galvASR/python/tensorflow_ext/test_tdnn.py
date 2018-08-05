import numpy as np
import tensorflow as tf

from galvASR.python.tensorflow_ext import tdnn


def test_tdnn():
  params = {
    'hidden_layer_dims': [512],
    'layer_splice_indices': [[-2, 0, 2], [-2, 0, 2]],
    'num_labels': 64,
    'learning_rate': 1e-5
  }
  model = tf.estimator.Estimator(tdnn.tdnn_model_fn_with_cross_entropy,
                                 params=params)

  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  x = np.random.normal(size=(time_steps, batch_size, input_feature_size)).astype(np.float32)
  y = np.random.randint(0, params['num_labels'],
                        size=(time_steps, batch_size)).astype(np.int32)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x=x, y=y,
                                                      batch_size=time_steps,
                                                      num_epochs=1,
                                                      shuffle=False)

  # How do estimators call tf.global_variables_initializer()?
  # model.train(input_fn=train_input_fn)
  model.predict(input_fn=train_input_fn)


# def test_tdnn_graph_creation():
#   batch_size = 16
#   time_steps = 20
#   input_feature_size = 40
#   features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
#   params = {
#     'hidden_layer_dims': [512],
#     'layer_splice_indices': [[-2, 0, 2], [-2, 0, 2]],
#     'num_labels': 64,
#     'learning_rate': 1e-5
#   }
#   tdnn.dynamic_tdnn(features, params)
#   with tf.Session() as sess:
#     writer = tf.summary.FileWriter("abc", sess.graph)
#     writer.close()


# def test_tdnn_gradients():
#   batch_size = 16
#   time_steps = 20
#   input_feature_size = 40
#   features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
#   params = {
#     'hidden_layer_dims': [512],
#     'layer_splice_indices': [[-2, 0, 2], [-2, 0, 2]],
#     'num_labels': 64,
#     'learning_rate': 1e-5
#   }
#   logits = tdnn.dynamic_tdnn(features, params)
#   labels = np.random.randint(0, params['num_labels'],
#                              size=(time_steps, batch_size)).astype(np.int32)
#   one_hot_labels = tf.one_hot(labels, params['num_labels'])
#   loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
#   # optimizer = tf.train.AdagradOptimizer(params['learning_rate'])
#   # train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
#   # grads = tf.gradients(loss, tf.trainable_variables())
#   with tf.Session() as sess:
#     writer = tf.summary.FileWriter("abc", sess.graph)
#     writer.close()


def test_small_tdnn_gradients():
  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
  params = {
    'hidden_layer_dims': [],
    'layer_splice_indices': [[-2, 0, 2]],
    'num_labels': 64,
    'learning_rate': 1e-5
  }
  logits = tdnn.dynamic_tdnn(features, params)
  labels = np.random.randint(0, params['num_labels'],
                             size=(time_steps, batch_size)).astype(np.int32)
  one_hot_labels = tf.one_hot(labels, params['num_labels'])
  loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
  # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,
  #                                                  logits=logits)
  # optimizer = tf.train.AdagradOptimizer(params['learning_rate'])
  # print("Gradients: ", optimizer.compute_gradients(loss))
  # train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
  # grads = tf.gradients(loss, tf.trainable_variables())

  from tensorflow.python.framework.ops import get_gradient_function
  for op in tf.get_default_graph().get_operations():
    if get_gradient_function(op) is None:
      print("Offending op: {}".format(op))

  with tf.Session() as sess:
    writer = tf.summary.FileWriter("abc_simple", sess.graph)
    writer.close()
