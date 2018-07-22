from functools import partial

import tensorflow as tf

# Note that this unfortunately requires that timesteps by run in
# sequence. loop_vars=1 necessarily
def _body(t, next_layer_ta, previous_layer, hidden_layer_dim, splice_indices, layer_index):
  # See if there's a way to get a strided tensor instead, and do a
  # batched matrix multiply, instead of doing this for loop.
  assert(isinstance(splice_indices, list))
  intermediate = []
  with tf.variable_scope('tdnn_{}'.format(layer_index), reuse=tf.AUTO_REUSE):
    for i, splice_index in enumerate(splice_indices):
      # TODO: tf.Variable vs tf.get_variable?
      # TODO: Update to python 3.6 for format strings
      W_i = tf.get_variable('W_{i}'.format(i=i), shape=[previous_layer.shape[2], hidden_layer_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
      intermediate.append(previous_layer[t + splice_index, :, :] @ W_i)
    next_layer_ta.write(t, tf.reduce_sum(intermediate, axis=0))
    # next_layer_ta[t, :, :] = tf.reduce_sum(intermediate, axis=1)
    t += 1
    return t, next_layer_ta, previous_layer


def tdnn_model_fn_with_cross_entropy(features, labels, mode, params):
  assert features.shape.ndims == 3  # T x B x D
  assert len(params['layer_splice_indices']) == len(params['hidden_layer_dims']) + 1

  if mode in {tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL}:
    assert labels.shape.ndims == 2  # T x B
    features.shape[0].assert_is_compatible_with(features.shape[0])
    features.shape[1].assert_is_compatible_with(features.shape[1])

  time_steps = tf.shape(features)[0]
  # batch_size = tf.shape(features)[1]
  const_batch_size = features.shape[1]

  # input_feature_dim = tf.shape(features)[2]
  # previous_layer_ta = tf.TensorArray(dtype=tf.float32,
  #                                    size=time_steps,
  #                                    element_shape=[batch_size, input_feature_dim],
  #                                    tensor_array_name="tdnn/input_features")
  # Do we need the input to be a TensorArray? I don't think so. In
  # fact, it may be better if it's not.
  # previous_layer_ta = previous_layer_ta.unstack(features)
  previous_layer = features

  hidden_layer_dims = params['hidden_layer_dims'] + [params['num_labels']]
  layer_splice_indices = params['layer_splice_indices']
  for i, (splice_indices, hidden_layer_dim) in enumerate(zip(layer_splice_indices,
                                                             hidden_layer_dims)):
    next_layer_ta = tf.TensorArray(dtype=tf.float32,
                                   size=time_steps,
                                   element_shape=tf.TensorShape([const_batch_size, hidden_layer_dim]),
                                   tensor_array_name="tdnn_{i}".format(i=i),
                                   clear_after_read=False)
    # next_layer = tf.get_variable('tdnn_{i}'.format(i=i),
    #                              initializer=tf.zeros(next_layer_shape),
    #                              trainable=False)

    assert(isinstance(splice_indices, list))
    if len(splice_indices) != len(set(splice_indices)):
      raise ValueError("Splice indices {splice_indices} at layer {} are not unique!".
                       format(splice_indices, i))

    t = tf.constant(0)
    specialized_body = partial(_body, hidden_layer_dim=hidden_layer_dim,
                               splice_indices=splice_indices, layer_index=i)
    with tf.name_scope("layer_{i}".format(i=i)):
      _, next_layer_ta, _ = tf.while_loop(lambda t, *_: t < time_steps,
                                          specialized_body, [t, next_layer_ta, previous_layer])
    next_layer = next_layer_ta.stack()
    if i == len(params['layer_splice_indices']) - 1:
      logits = next_layer
      # TODO: Check whether this will reduce over only the last
      # dimension
      one_hot_labels = tf.one_hot(labels, params['num_labels'])
      loss = tf.losses.softmax_cross_entropy(one_hot_labels, next_layer)
    else:
      previous_layer = tf.nn.relu(next_layer)
      # previous_layer_ta = tf.TensorArray(dtype=tf.float32,
      #                                    size=time_steps,
      #                                    element_shape=[batch_size, hidden_layer_dim],
      #                                    tensor_array_name="tdnn/hidden_features_{i}".format(i=i))
      # previous_layer_ta = previous_layer_ta.unstack(next_layer)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(params['learning_rate'])

    return tf.estimator.EstimatorSpec(
      mode=tf.estimator.ModeKeys.TRAIN,
      loss=loss,
      train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
