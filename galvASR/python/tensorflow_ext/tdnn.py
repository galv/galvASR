import tensorflow as tf

def conv1d_tdnn(features, labels, mode, params):
  assert features.shape.ndims == 3, features.shape.ndims  # B x T x D
  assert len(params['num_splices']) == len(params['hidden_layer_dims']) + 1

  if mode in {tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL}:
    assert labels.shape.ndims == 2  # B x T
    features.shape[0].assert_is_compatible_with(features.shape[0])
    features.shape[1].assert_is_compatible_with(features.shape[1])

  batch_size_tensor = tf.cast(tf.shape(features)[0], tf.float32)
  previous_layer = features

  hidden_layer_dims = params['hidden_layer_dims'] + [params['num_labels']]
  num_splices = params['num_splices']
  dilations = params['dilations']
  for i, (hidden_layer_dim, splice_range, dilation) in enumerate(
      zip(hidden_layer_dims, num_splices, dilations)):
    activation = "linear" if i == len(num_splices) - 1 else "relu"
    next_layer = tf.layers.conv1d(
        previous_layer,
        hidden_layer_dim,
        splice_range,
        padding="valid",
        dilation_rate=dilation,
        activation=activation,
        name="layer{i}".format(i=i))

    previous_layer = next_layer

  logits = previous_layer

  if mode == tf.estimator.ModeKeys.TRAIN:
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    optimizer = tf.train.AdagradOptimizer(params['learning_rate'])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(loss, tf.train.get_or_create_global_step()))
  elif mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        "classes": tf.argmax(logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  else:
    assert mode == tf.estimator.ModeKeys.EVAL
    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss)
