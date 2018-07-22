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
  model.train(input_fn=train_input_fn)
