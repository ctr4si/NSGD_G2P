from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from data_utils_g2p import prepare_g2p_data
from experiment import Experiment
from learning import train, test
from paths import *

""" Running info """
tf.app.flags.DEFINE_float("gpu_memory_fraction", 0.56, "Memory allocation rate")
tf.app.flags.DEFINE_boolean("model_parameter_saving", True, "Whether saving model parameters or not")
tf.app.flags.DEFINE_boolean("eval_only", False, "Run eval procedure only.")

# Only if 'load_pre_trained_model' is True or 'eval_only' is True, this argument is used.
tf.app.flags.DEFINE_integer("model_id_to_load", -1,
                            "Id of model with specific parameter setting. "
                            "If this value is not set, current model id will be used.")

""" Model architecture """
tf.app.flags.DEFINE_integer("embedding_size", 256, "Number of layers in the model.")

# conv structures using source sequence
tf.app.flags.DEFINE_integer("source_num_layers", 5, "Number of layers (or blocks) in the model.")
tf.app.flags.DEFINE_integer("source_filter_width", 3, "Filter width at 1d-conv")

# conv structures using target sequence
tf.app.flags.DEFINE_integer("target_num_layers", 5, "Number of layers (or blocks) in the model.")
tf.app.flags.DEFINE_integer("target_filter_width", 3, "Filter width at 1d-conv")

# conv structures for decoding
tf.app.flags.DEFINE_integer("decoding_num_layers", 5, "Number of layers (or blocks) in the model.")
tf.app.flags.DEFINE_integer("decoding_filter_width", 3, "Filter width at 1d-conv")

""" Learning algorithm """
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.98, "Learning rate decays by this much.")
tf.app.flags.DEFINE_integer("adapting_cycle_steps", 40,
                            "How many steps are needed to adapt learning rate.")
tf.app.flags.DEFINE_integer("adapting_queue_size", 3,
                            "Number of logs of previous avg losses that are used at learning rate adapting.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("num_epochs", 100, "How many training steps to do until stop training (0: no limit).")

""" etc """
# Debugging
tf.app.flags.DEFINE_boolean("inference_debugging", False, "Set to True for printing detailed predictions.")

# Only if 'inference_debugging' is True, this argument is used.
tf.app.flags.DEFINE_boolean("wrong_only_debugging", False,
                            "Set to True for applying debugging only to wrongly predicted instances.")

# Only if 'eval_only' is False, this argument is used.
tf.app.flags.DEFINE_boolean("summary_write", False, "Set to True for writing summary.")
tf.app.flags.DEFINE_integer("summary_step_cycle", 5, "How many steps needed for each summary writing")

# Regularization
tf.app.flags.DEFINE_float("weight_decay", 0.00001, "Weight decay regularization")
tf.app.flags.DEFINE_float("residual_reg_keep_prob", 0.8, "keep probability of dropout. Default: 0.8.")

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    train_sources, train_targets, valid_sources, valid_targets, test_sources, test_targets, \
    source_vocab, source_idx2char, target_vocab, target_idx2char = prepare_g2p_data(FLAGS)

    expeirment = Experiment(source_vocab, source_idx2char, target_vocab, target_idx2char, FLAGS, VERSION)

    if not FLAGS.eval_only:
        train(train_sources, train_targets, valid_sources, valid_targets, expeirment, "[TR]")
        ops.reset_default_graph()

    if FLAGS.model_parameter_saving or FLAGS.eval_only:
        test(valid_sources, valid_targets, test_sources, test_targets, expeirment, "[TE]")


if __name__ == "__main__":
    tf.app.run()
