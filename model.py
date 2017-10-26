__author__ = 'jjamjung'

from layer_methods import *

SRC_ENC = 'source_encoding'
TRG_ENC = 'target_encoding'
DEC = 'decoding'

DILATIONS = [1, 2, 4, 8, 16] * 5


def conv_seq2seq_model(source_seq, target_seq, is_training, experiment):
    num_layers, filter_shapes, filter_nums, conv_strides, conv_paddings = parameter_handling(experiment)

    # frequently used variables
    num_source_vocabs = experiment.num_source_vocabs.value
    num_target_vocabs = experiment.num_target_vocabs.value
    embedding_size = experiment.embedding_size.value
    weight_decay = experiment.weight_decay.value
    residual_reg_keep_prob = experiment.residual_reg_keep_prob.value

    with tf.variable_scope(SRC_ENC):
        source_seq = encode_sequence(source_seq, is_training,
                                     num_voca=num_source_vocabs,
                                     embedding_size=embedding_size,
                                     weight_decay=weight_decay,
                                     num_layers=num_layers[SRC_ENC],
                                     filter_shapes=filter_shapes[SRC_ENC],
                                     filter_nums=filter_nums[SRC_ENC],
                                     conv_strides=conv_strides[SRC_ENC],
                                     conv_paddings=conv_paddings[SRC_ENC],
                                     residual_reg_keep_prob=residual_reg_keep_prob)

    with tf.variable_scope(TRG_ENC):
        target_seq = encode_sequence(target_seq, is_training,
                                     num_voca=num_target_vocabs,
                                     embedding_size=embedding_size,
                                     weight_decay=weight_decay,
                                     num_layers=num_layers[TRG_ENC],
                                     filter_shapes=filter_shapes[TRG_ENC],
                                     filter_nums=filter_nums[TRG_ENC],
                                     conv_strides=conv_strides[TRG_ENC],
                                     conv_paddings=conv_paddings[TRG_ENC],
                                     residual_reg_keep_prob=residual_reg_keep_prob)

    encoded_seq = tf.concat([source_seq, target_seq], axis=2)

    with tf.variable_scope(DEC):
        logit_seq = decode_sequence(encoded_seq, is_training,
                                    num_target_voca=num_target_vocabs,
                                    weight_decay=weight_decay,
                                    num_layers=num_layers[DEC],
                                    filter_shapes=filter_shapes[DEC],
                                    filter_nums=filter_nums[DEC],
                                    conv_strides=conv_strides[DEC],
                                    conv_paddings=conv_paddings[DEC],
                                    residual_reg_keep_prob=residual_reg_keep_prob)

    return logit_seq


def encode_sequence(input_seq, is_training, num_voca, embedding_size, weight_decay,
                    num_layers, filter_shapes, filter_nums, conv_strides, conv_paddings,
                    residual_reg_keep_prob):
    encoded_seq = embedding(input_seq, num_voca, embedding_size, weight_decay)

    for conv_layer_idx in range(num_layers):
        with tf.variable_scope("conv-%d" % (conv_layer_idx + 1)):
            encoded_seq = residual_block(encoded_seq, is_training,
                                         filter_shape=filter_shapes[conv_layer_idx],
                                         num_filters_out=filter_nums[conv_layer_idx],
                                         strides=conv_strides[conv_layer_idx],
                                         padding=conv_paddings[conv_layer_idx],
                                         weight_decay=weight_decay,
                                         dropout_keep_prob=residual_reg_keep_prob)
    return encoded_seq


def decode_sequence(encoded_seq, is_training, num_target_voca, weight_decay,
                    num_layers, filter_shapes, filter_nums, conv_strides, conv_paddings,
                    residual_reg_keep_prob):
    for conv_layer_idx in range(num_layers):
        with tf.variable_scope("conv-%d" % (conv_layer_idx + 1)):
            encoded_seq = residual_block(encoded_seq, is_training,
                                         filter_shape=filter_shapes[conv_layer_idx],
                                         num_filters_out=filter_nums[conv_layer_idx],
                                         strides=conv_strides[conv_layer_idx],
                                         padding=conv_paddings[conv_layer_idx],
                                         weight_decay=weight_decay,
                                         dropout_keep_prob=residual_reg_keep_prob)

    with tf.variable_scope("output-conv-1"):
        encoded_seq = layer_normalization(encoded_seq)

        logit_seq = conv(encoded_seq,
                         filter_shape=[1],
                         num_filters_out=num_target_voca,
                         strides=1,
                         padding='SAME',
                         weight_decay=weight_decay)

    return logit_seq


def parameter_handling(parameters):
    param_keys = [SRC_ENC, TRG_ENC, DEC]
    num_layers = {SRC_ENC: parameters.source_num_layers.value,
                  TRG_ENC: parameters.target_num_layers.value,
                  DEC: parameters.decoding_num_layers.value}
    filter_widths = {SRC_ENC: parameters.source_filter_width.value,
                     TRG_ENC: parameters.target_filter_width.value,
                     DEC: parameters.decoding_filter_width.value}

    """ convolutional layers """
    conv_paddings = {key: ['SAME'] * num_layers[key] for key in param_keys}
    filter_shapes = {key: [[filter_widths[key]]] * num_layers[key] for key in param_keys}
    filter_nums = {SRC_ENC: [parameters.embedding_size.value] * num_layers[SRC_ENC],
                   TRG_ENC: [parameters.embedding_size.value] * num_layers[TRG_ENC],
                   DEC: [parameters.embedding_size.value * 2] * num_layers[DEC]}
    conv_strides = {key: [1] * num_layers[key] for key in param_keys}

    return num_layers, filter_shapes, filter_nums, conv_strides, conv_paddings
