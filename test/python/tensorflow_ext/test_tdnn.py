import numpy as np
import tensorflow as tf

from galvASR.python.tensorflow_ext import tdnn

def test_tdnn_conv1d():
  params = {
      'hidden_layer_dims': [512],
      'num_splices': [3, 3],
      'dilations': [2, 2],
      'num_labels': 64,
      'learning_rate': 1e-5
  }
  model = tf.estimator.Estimator(tdnn.conv1d_tdnn, params=params)

  total_context = sum([(num_splice - 1) * dilation for num_splice, dilation in
                       zip(params['num_splices'], params['dilations'])])

  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  x = (np.random.normal(
      size=(batch_size, time_steps + total_context,
            input_feature_size)).astype(np.float32))
  y = np.random.randint(
      0, params['num_labels'], size=(batch_size, time_steps)).astype(np.int32)
  # dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(time_steps)
  # iterator = dataset.make_one_shot_iterator()
  # input_fn = lambda : iterator.get_next()
  input_fn = tf.estimator.inputs.numpy_input_fn(
      x=x, y=y, batch_size=time_steps, num_epochs=1, shuffle=False)

  # How do estimators call tf.global_variables_initializer()?
  model.train(input_fn=input_fn)
  for prediction in model.predict(input_fn=input_fn):
    print(prediction)


def test_rnn_gradients():
  batch_size = 16
  time_steps = 20
  input_feature_size = 40
  hidden_layer_dims = [512, 512]
  num_labels = 64
  g = tf.Graph()
  with g.as_default():
    features = tf.placeholder(tf.float32,
                              (time_steps, batch_size, input_feature_size))
    labels = tf.placeholder(tf.int32, shape=(time_steps, batch_size))
    rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in hidden_layer_dims]
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
    outputs, _ = tf.nn.dynamic_rnn(
        cell=multi_rnn_cell,
        inputs=features,
        time_major=True,
        dtype=features.dtype)
    W = tf.get_variable("W_hidden_to_output",
                        [hidden_layer_dims[-1], num_labels])
    # This version of logits is missing the time dimension. Ugh.
    logits = tf.tensordot(outputs, W, 1)
    print(logits.shape)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    grads = tf.gradients(loss,
                         g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    assert len(grads) > 0
