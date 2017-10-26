__author__ = 'jjamjung'

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

UPDATE_OPS_COLLECTION = "update_ops"
WEIGHT_VAR = "weight"
EMBEDDING_VAR = "embedding"


def embedding(x, unique_input_num, vector_size, weight_decay):
    embeddings = _get_variable(EMBEDDING_VAR, [unique_input_num, vector_size],
                               initializer=variance_scaling_initializer(),
                               weight_decay=weight_decay)
    embed = tf.nn.embedding_lookup(embeddings, x)
    return embed


def dropout(x, keep_prob, is_training, seed=None):
    def __apply_dropout():
        d = tf.nn.dropout(x, keep_prob, seed=seed)
        return d

    return tf.cond(is_training, __apply_dropout, lambda: x)


def layer_normalization(inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = tf.divide(tf.subtract(inputs, mean), (tf.pow(tf.add(variance, epsilon), tf.constant(.5))))
        outputs = tf.add(tf.multiply(gamma, normalized), beta)

    return outputs


def conv(x, filter_shape, num_filters_out, strides, padding, weight_decay=0.00001):
    assert type(filter_shape) == list
    x_shape = x.get_shape().as_list()
    total_shape = filter_shape + [x_shape[-1], num_filters_out]
    w = _get_variable(WEIGHT_VAR, shape=total_shape,
                      initializer=variance_scaling_initializer(),
                      weight_decay=weight_decay)

    if len(x_shape) == 4:
        return tf.nn.conv2d(x, w, strides, padding=padding)
    else:
        return tf.nn.conv1d(x, w, strides, padding=padding)


def residual_block(x, is_training, filter_shape, num_filters_out, strides, padding,
                   weight_decay=0.00001, dropout_keep_prob=0.8):
    x_shape = x.get_shape().as_list()
    num_filters_in = x_shape[-1]

    # TODO: shorcut shape adapting
    shortcut = x

    x = dropout(x, dropout_keep_prob, is_training)

    for unit_name in ['a', 'b']:
        with tf.variable_scope(unit_name):
            x = tf.nn.relu(x)

            x = layer_normalization(x)

            x = conv(x, filter_shape, num_filters_out, strides, padding, weight_decay=weight_decay)

    with tf.variable_scope("short_cut"):
        if num_filters_in != num_filters_out:
            x_shape = shortcut.get_shape().as_list()
            if len(x_shape) == 3:
                shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [(num_filters_out - num_filters_in) // 2] * 2])
            else:
                shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [(num_filters_out - num_filters_in) // 2] * 2])
        result = tf.add(x, shortcut)

    return result


def _get_variable(name, shape, initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True,
                  collections=None):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None

    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           trainable=trainable,
                           collections=collections)
