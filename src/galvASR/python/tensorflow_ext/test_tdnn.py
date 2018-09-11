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


def test_tdnn_graph_creation():
  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  params = {
    'hidden_layer_dims': [512],
    'layer_splice_indices': [[-2, 0, 2], [-2, 0, 2]],
    'num_labels': 64,
    'learning_rate': 1e-5
  }
  g = tf.Graph()
  with g.as_default():
    features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
    labels = tf.placeholder(tf.int32, shape=(time_steps, batch_size))
    logits = tdnn.dynamic_tdnn(features, params)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    grads = tf.gradients(loss, g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    print("Gradients", grads)
    print("Variables", g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  writer = tf.summary.FileWriter("abc", g)
  writer.close()


def test_rnn_gradients():
  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  hidden_layer_dims = [512, 512]
  num_labels = 64
  g = tf.Graph()
  with g.as_default():
    features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
    labels = tf.placeholder(tf.int32, shape=(time_steps, batch_size))
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden_layer_dims]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, _ = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                   inputs=features,
                                   time_major=True,
                                   dtype=features.dtype)
    W = tf.get_variable("W_hidden_to_output", [hidden_layer_dims[-1], num_labels])
    # This version of logits is missing the time dimension. Ugh.
    logits = tf.tensordot(outputs, W, 1)
    print(logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    grads = tf.gradients(loss, g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    assert len(grads) > 0


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


# def test_small_tdnn_gradients():
#   batch_size = 16
#   time_steps = 20
#   input_feature_size = 40
#   features = tf.placeholder(tf.float32, (time_steps, batch_size, input_feature_size))
#   params = {
#     'hidden_layer_dims': [],
#     'layer_splice_indices': [[-2, 0, 2]],
#     'num_labels': 64,
#     'learning_rate': 1e-5
#   }
#   logits = tdnn.dynamic_tdnn(features, params)
#   labels = np.random.randint(0, params['num_labels'],
#                              size=(time_steps, batch_size)).astype(np.int32)
#   one_hot_labels = tf.one_hot(labels, params['num_labels'])
#   loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits)
#   # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_labels,
#   #                                                  logits=logits)
#   # optimizer = tf.train.AdagradOptimizer(params['learning_rate'])
#   # print("Gradients: ", optimizer.compute_gradients(loss))
#   # train_op = optimizer.minimize(loss, tf.train.get_or_create_global_step())
#   # grads = tf.gradients(loss, tf.trainable_variables())

#   from tensorflow.python.framework.ops import get_gradient_function
#   for op in tf.get_default_graph().get_operations():
#     if get_gradient_function(op) is None:
#       print("Offending op: {}".format(op))

#   with tf.Session() as sess:
#     writer = tf.summary.FileWriter("abc_simple", sess.graph)
#     writer.close()
