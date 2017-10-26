# -*- coding: utf-8 -*-
__author__ = 'jjamjung'

import math
import os
import shutil
import time

from etc_methods import *
from layer_methods import *
from model import conv_seq2seq_model
from paths import UNK, BLK, PAD, EOS


def train(tr_sources, tr_targets, va_sources, va_targets, experiment, print_tag):
    experiment.print_all_params()

    # frequently used variables
    max_len = experiment.max_len.value
    batch_size = experiment.batch_size.value
    num_target_vocabs = experiment.num_target_vocabs.value
    dataset_size = len(tr_targets)

    """ ========================== Preparing ========================== """
    # Placeholders
    x = tf.placeholder(tf.int32, shape=(None, max_len))
    y = tf.placeholder(tf.int32, shape=(None, max_len))
    gt = tf.placeholder(tf.int32, shape=(None, max_len))
    gt_oh = tf.one_hot(gt, depth=num_target_vocabs, on_value=1., axis=-1)
    is_training = tf.placeholder(tf.bool)

    # Create graph
    logits = conv_seq2seq_model(x, y, is_training, experiment)

    # Cross_entropy, loss, prediction and error rates are defined.
    prediction = tf.nn.softmax(logits)
    cross_entropy = output_cross_entropy(logits, gt_oh)
    loss = output_loss(cross_entropy)

    # Learning rate decaying
    learning_rate = tf.Variable(experiment.learning_rate.value,
                                trainable=False, name='learning_rate',
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'TO_SAVE'],
                                dtype=tf.float32)
    decay_factor = tf.constant(experiment.learning_rate_decay_factor.value, dtype=tf.float32)
    learning_rate_update = tf.assign(learning_rate, tf.multiply(learning_rate, decay_factor))

    # Training step
    train_op, grads_op, grad_var_list, step_var = make_train_op(learning_rate, loss)

    # Create new directories for saving model experiment and summaries
    summary_dir = experiment.initialize_directories()

    # Summary ops
    if experiment.summary_write.value:
        tf.summary.scalar("loss", loss)
        for gv in grad_var_list:
            tf.summary.scalar(gv.name, grads_op[gv.name])
    merged_summary_ops = tf.summary.merge_all() if experiment.summary_write.value else None

    """ ========================== MAIN LOOP ========================== """
    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=experiment.gpu_memory_fraction.value),
                allow_soft_placement=True)) as sess:

        # Variables initialization
        sess.run(tf.global_variables_initializer())

        # Saver
        saver = make_saver(print_tag, experiment)

        # Summary writer
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)

        # For time cost check
        start_time = time.time()
        epoch_start_time, epoch_duration_time = time.time(), 0

        # For learning rate adapting
        avg_losses = 0.
        losses_queue = []

        # model selection
        best_score = np.inf
        best_step = 0

        curr_index_in_epoch = 0  # for batch data sampling
        total_step_num = experiment.get_total_step_num(dataset_size, batch_size)
        tr_sources, tr_targets = dataset_shuffling(tr_sources, tr_targets)
        for step in range(total_step_num)[sess.run(step_var):]:

            # Batch sampling
            batch_x, batch_gt = get_batch(tr_sources, tr_targets, curr_index_in_epoch, batch_size)
            curr_index_in_epoch += batch_size

            # Each epoch re-shuffling
            if curr_index_in_epoch + batch_size >= dataset_size:
                tr_sources, tr_targets = dataset_shuffling(tr_sources, tr_targets)
                curr_index_in_epoch = 0

            # Batch training : multi-steps or single-step learning
            avg_losses += batch_training(batch_x, batch_gt,
                                         ph_x=x, ph_y=y, ph_gt=gt, ph_is_training=is_training,
                                         train_op=train_op, loss_op=loss, summary_op=merged_summary_ops,
                                         session=sess, summary_writer=summary_writer,
                                         experiment=experiment, current_step=step)

            # Learning rate adapting
            if experiment.time_to_check_loss(step):
                avg_losses /= experiment.adapting_cycle_steps.value
                current_learning_rate = learning_rate.eval(session=sess)

                str_to_print = \
                    "Epoch %.2f, Step %d, LR: %.6f, AVG loss for %d-steps: %.3f" % \
                    (experiment.get_curr_epoch_in_float(step, dataset_size),
                     step, current_learning_rate, experiment.adapting_cycle_steps.value, avg_losses)

                if len(losses_queue) > 0:
                    if experiment.learning_rate_decay_factor.value < 1.0:
                        if avg_losses > max(losses_queue):
                            sess.run(learning_rate_update)
                            str_to_print += ", Learning rate decaying..."

                losses_queue.append(avg_losses)
                if len(losses_queue) > experiment.adapting_queue_size.value:
                    del losses_queue[0]
                avg_losses = 0.
                print_with_tag(str_to_print, print_tag, 1)

            if experiment.time_to_save_model(step, total_step_num, dataset_size):
                curr_epoch_in_int = experiment.get_curr_epoch_in_int(step, dataset_size)

                # Evaluation
                va_results = evaluation(x=va_sources, gt=va_targets, ph_x=x, ph_y=y, ph_is_training=is_training,
                                        prediction_op=prediction, session=sess,
                                        experiment=experiment, print_tag=print_tag)

                epoch_duration_time = time.time() - epoch_start_time
                epoch_start_time = time.time()

                str_to_print = \
                    "Epoch %.2f, Step %d, epoch_time_cost: %d" % \
                    (experiment.get_curr_epoch_in_float(step, dataset_size), step, epoch_duration_time)

                str_to_print += ", " + eval_result_to_string(va_results, "VA")

                # Model improvement
                is_improved = va_results[1] < best_score
                if is_improved:
                    str_to_print += ", IMPROVED! (Step: %d, Error: %.3f)" % (best_step, best_score * 100)
                    experiment.update_va_performances(va_results[0], va_results[1], curr_epoch_in_int)

                    best_score = va_results[1]
                    best_step = step

                    # Save the model checkpoint.
                    if experiment.model_parameter_saving.value:
                        checkpoint_path = \
                            os.path.join(
                                experiment.get_checkpoint_path(model_idx=experiment.model_id.value,
                                                               epoch=curr_epoch_in_int),
                                "model.ckpt")
                        saver.save(sess, checkpoint_path, global_step=step)

                # Print evaluation results
                print_with_tag(str_to_print, print_tag, 2)

    training_time = time.time() - start_time
    print_with_tag("", print_tag)
    print_with_tag("Training complete -- epoch limit reached", print_tag, 1)
    print_with_tag("Model: %d, Epoch: %d, VA_PER: %.3f%%, VA_WER: %.3f%%" %
                   (experiment.model_id.value, experiment.best_epoch.value,
                    experiment.va_per.value * 100, experiment.va_wer.value * 100),
                   print_tag, 1)
    print_with_tag("The session ran for %.2fm" % (training_time / 60.), print_tag, 1)
    print_with_tag("", print_tag, 1)
    print_with_tag("", print_tag, 1)

    experiment.finish_training()


def test(va_x, va_y, te_x, te_y, experiment, print_tag):
    experiment.print_all_params()

    # frequently used variables
    max_len = experiment.max_len.value

    # Placeholders
    x = tf.placeholder(tf.int32, shape=(None, max_len))
    y = tf.placeholder(tf.int32, shape=(None, max_len))
    is_training = tf.placeholder(tf.bool)

    # Create graph
    logits = conv_seq2seq_model(x, y, is_training, experiment)

    # Cross_entropy, loss, prediction and error rates are defined.
    prediction = tf.nn.softmax(logits)

    with tf.Session(
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=experiment.gpu_memory_fraction.value),
                allow_soft_placement=True)) as sess:
        # Variables initialization
        sess.run(tf.global_variables_initializer())

        # Saver
        saver = make_saver(print_tag, experiment)

        model_dir = os.path.join(experiment.checkpoint_base_path, str(experiment.model_id.value))
        model_epoch_lists = sorted([int(me) for me in os.listdir(model_dir)])

        best_score = np.inf
        top_scores, top_epoch_index = np.array([np.inf] * 5), np.zeros(5)
        whole_dirs = []

        for me in model_epoch_lists:
            # Restore variables
            verbose, ckpt_dir = restore_variables(experiment, saver, sess, print_tag, epoch=me, for_eval=True)
            whole_dirs.append(ckpt_dir)

            if verbose == "SKIP":
                continue
            elif verbose == "FAIL":
                return

            # Evaluation
            va_results = evaluation(x=va_x, gt=va_y, ph_x=x, ph_y=y, ph_is_training=is_training,
                                    prediction_op=prediction, session=sess, experiment=experiment, print_tag=print_tag)

            te_results = evaluation(x=te_x, gt=te_y, ph_x=x, ph_y=y, ph_is_training=is_training,
                                    prediction_op=prediction, session=sess, experiment=experiment, print_tag=print_tag)

            # Print results
            print_with_tag("VA PER: %.3f%%, VA WER: %.3f%%, TE PER: %.3f%%, TE WER: %.3f%%" %
                           (va_results[0] * 100, va_results[1] * 100, te_results[0] * 100, te_results[1] * 100),
                           print_tag, 2)
            print_with_tag("", print_tag)

            is_improved = va_results[1] < best_score
            if is_improved:
                experiment.update_va_performances(va_results[0], va_results[1], me)
                experiment.update_te_performances(te_results[0], te_results[1])
                best_score = va_results[1]

            is_top5 = va_results[1] < top_scores[-1]
            if is_top5:
                top_scores[-1] = va_results[1]
                top_epoch_index[-1] = len(whole_dirs) - 1

                sorted_args = np.argsort(top_scores)
                top_scores = top_scores[sorted_args]
                top_epoch_index = top_epoch_index[sorted_args]

        print_with_tag("Selected model performances", print_tag, 1)
        experiment.print_performances()
        print_with_tag("", print_tag, 1)
        print_with_tag("", print_tag, 1)

        for i in range(len(whole_dirs)):
            if i not in top_epoch_index:
                shutil.rmtree(whole_dirs[i])

        experiment.finish_test()


def evaluation(x, gt, ph_x, ph_y, ph_is_training, prediction_op, session, experiment, print_tag):
    eval_results, str_logs = \
        evaluate_greedy_inference(x, gt,
                                  ph_x=ph_x,
                                  ph_y=ph_y,
                                  ph_is_training=ph_is_training,
                                  prediction_op=prediction_op,
                                  session=session,
                                  batch_size=experiment.batch_size.value,
                                  debugging=experiment.inference_debugging.value,
                                  wrong_only=experiment.wrong_only_debugging.value,
                                  experiment=experiment)

    if len(str_logs) > 0:
        print_with_tag("", print_tag, 2)
    for log in str_logs:
        print_with_tag(log, print_tag, 2)

    return eval_results


def evaluate_greedy_inference(x, gt,
                              ph_x, ph_y, ph_is_training,
                              prediction_op,
                              session,
                              batch_size,
                              debugging=False,
                              wrong_only=False,
                              experiment=None):
    total_step_num = int(math.ceil(float(gt.shape[0]) / batch_size))
    total_phoneme_error, total_word_error = 0., 0.
    total_phonemes, total_words = 0, 0
    curr_index_in_epoch = 0

    edit_distances = []

    inference_log_list = []
    true_false_log_list = []
    ground_truth_list = []
    source_sequence_list = []
    seq_prediction_list = []
    string_list = []

    blk_id = experiment.get_target_blk_id()

    for step in xrange(total_step_num):

        batch_x, batch_gt = get_batch(x, gt, curr_index_in_epoch, batch_size)
        curr_index_in_epoch += batch_size

        batch_index_array = np.arange(len(batch_gt))

        # sequence of predictions made by decoding procedure, shape: (# of instances, input length)
        batch_prediction = np.zeros_like(batch_gt)

        # equal shape with batch_prediction, 1 at blank positions and 0 at others.
        batch_already_predicted_or_not = (batch_prediction == blk_id).astype(float)

        batch_inference_logs = []
        batch_sequence_prediction_logs = []
        batch_true_false_logs = []

        for pred_step in xrange(experiment.max_len.value):
            # feed updated batch_prediction to ph_y
            single_step_prediction = session.run(prediction_op,
                                                 feed_dict={ph_x: batch_x,
                                                            ph_y: batch_prediction,
                                                            ph_is_training: False})

            # sequence of confidence values, shape: (# of instances, input length)
            confidence_sequence = np.max(single_step_prediction, axis=2)

            # if it is not the first prediction_op step,
            # then confidence values at previous predicted indices are replaced with zero.
            confidence_sequence = confidence_sequence * batch_already_predicted_or_not

            # sequence of predictions, shape: (# of instances, input length)
            prediction_sequence = np.argmax(single_step_prediction, axis=2)
            batch_sequence_prediction_logs.append(prediction_sequence)

            # shape: (# of instances)
            batch_predicted_index = np.argmax(confidence_sequence, axis=1)

            batch_tf = None
            if debugging:
                batch_tf = \
                    prediction_sequence[batch_index_array, batch_predicted_index] == \
                    batch_gt[batch_index_array, batch_predicted_index]

            # only prediction_op at the selected index is assigned
            batch_prediction[batch_index_array, batch_predicted_index] = \
                prediction_sequence[batch_index_array, batch_predicted_index]

            # update already predicted or not table
            batch_already_predicted_or_not[batch_index_array, batch_predicted_index] = 0.

            if debugging:
                batch_true_false_logs.append(batch_tf)
                batch_inference_logs.append(np.copy(batch_prediction))

        if debugging:
            # inference_logs, list of np.ndarray
            ilog = batch_inference_logs[-1]

            batch_inference_logs = np.swapaxes(np.array(batch_inference_logs), axis1=0, axis2=1)
            # inference_logs, shape: (batch_size, pred_step, sequence)

            batch_true_false_logs = np.swapaxes(np.array(batch_true_false_logs), axis1=0, axis2=1)
            # true_false_logs, shape: (batch_size, pred_step)

            batch_sequence_prediction_logs = np.swapaxes(np.array(batch_sequence_prediction_logs), axis1=0, axis2=1)

            if wrong_only:
                debug_index = (ilog != batch_gt).any(axis=1)
                batch_inference_logs = batch_inference_logs[debug_index, :, :]
                batch_true_false_logs = batch_true_false_logs[debug_index, :]
                ground_truth_list.append(batch_gt[debug_index, :])
                source_sequence_list.append(batch_x[debug_index, :])
                batch_sequence_prediction_logs = batch_sequence_prediction_logs[debug_index, :, :]
            else:
                ground_truth_list.append(batch_gt)
                source_sequence_list.append(batch_x[:, :])

            inference_log_list.append(batch_inference_logs)
            true_false_log_list.append(batch_true_false_logs)
            seq_prediction_list.append(batch_sequence_prediction_logs)

        edit_distances += compute_batch_edit_distances_for_decoding(source_arr=batch_x,
                                                                    prediction_arr=batch_prediction,
                                                                    ground_truth_arr=batch_gt,
                                                                    experiment=experiment)

    if debugging:
        inference_logs = np.concatenate(inference_log_list)
        true_false_logs = np.concatenate(true_false_log_list)
        ground_truths = np.concatenate(ground_truth_list)
        source_sequences = np.concatenate(source_sequence_list)
        seq_predictions = np.concatenate(seq_prediction_list)

        random_order = np.arange(len(inference_logs))
        np.random.shuffle(random_order)
        random_order = random_order[:10]

        inference_logs = inference_logs[random_order]
        true_false_logs = true_false_logs[random_order]
        ground_truths = ground_truths[random_order]
        source_sequences = source_sequences[random_order]
        seq_predictions = seq_predictions[random_order]

        if experiment is not None:
            to_parse = {UNK, PAD, BLK, EOS}
            source_vocab = {k: str(v)[1:] if str(v) in to_parse else str(v) for k, v in
                            experiment.source_idx2char.value.items()}
            source_vocab = {k: v + ('_' * (3 - len(v))) for k, v in source_vocab.items()}
            source_vocab = {k: '***' if v == 'PAD' else v for k, v in source_vocab.items()}
            source_vocab = {k: '___' if v == 'BLK' else v for k, v in source_vocab.items()}
            target_vocab = {k: str(v)[1:] if str(v) in to_parse else str(v) for k, v in
                            experiment.target_idx2char.value.items()}
            target_vocab = {k: v + ('_' * (3 - len(v))) for k, v in target_vocab.items()}
            target_vocab = {k: '***' if v == 'PAD' else v for k, v in target_vocab.items()}
            target_vocab = {k: '___' if v == 'BLK' else v for k, v in target_vocab.items()}

            inference_logs = ids_to_symbols(inference_logs, target_vocab)
            ground_truths = ids_to_symbols(ground_truths, target_vocab)
            source_sequences = ids_to_symbols(source_sequences, source_vocab)
            seq_predictions = ids_to_symbols(seq_predictions, target_vocab)
        else:
            inference_logs = ids_to_strings(inference_logs)
            ground_truths = ids_to_strings(ground_truths)
            source_sequences = ids_to_strings(source_sequences)
            seq_predictions = ids_to_strings(seq_predictions)

        for i in range(len(inference_logs)):
            for j in range(len(inference_logs[i])):
                string_list.append(
                    "%s: %s" % ('T' if true_false_logs[i][j] else 'F', ' '.join(inference_logs[i][j])) +
                    "\t%s" % (' '.join(seq_predictions[i][j]))
                )
            string_list.append('O: %s' % ' '.join(ground_truths[i]))
            string_list.append('W: %s' % ' '.join(source_sequences[i]))
            string_list.append("")

    edit_distances.sort()

    # Aggregate the edit distances for each word
    word_to_edit = {}
    for edit_distance in edit_distances:
        word, distance, target_seq_len = edit_distance
        word = tuple(word)
        if word in word_to_edit:
            word_to_edit[word].append((distance, target_seq_len))
        else:
            word_to_edit[word] = [(distance, target_seq_len)]

    total_words = len(word_to_edit)
    for word in word_to_edit:
        # Pick the ground truth that's closest to output since their can be
        # multiple pronunciations
        distance, target_seq_len = min(word_to_edit[word])
        if distance != 0:
            total_word_error += 1
            total_phoneme_error += distance
        total_phonemes += target_seq_len

    total_phoneme_error /= total_phonemes
    total_word_error /= total_words
    return (total_phoneme_error, total_word_error), string_list


def batch_training(batch_x, batch_gt,
                   ph_x, ph_y, ph_gt, ph_is_training,
                   train_op, loss_op, summary_op,
                   session, summary_writer, experiment, current_step):
    ops_to_run = [train_op, loss_op]
    summary_write = experiment.time_to_summarize(current_step)
    if summary_write:
        ops_to_run.append(summary_op)

    batch_results = \
        session.run(ops_to_run,
                    feed_dict={ph_x: batch_x,
                               ph_y: prepare_y(batch_gt, drop_whole_seq=False),
                               ph_gt: batch_gt,
                               ph_is_training: True})

    if summary_write:
        summary_writer.add_summary(batch_results[-1], current_step)

    return batch_results[1]


def restore_variables(experiment, saver, session, print_tag, epoch, for_eval=True):
    # Restore variables
    if for_eval:
        target_model_id = experiment.model_id.value
    else:
        target_model_id = experiment.model_id_to_load.value

    ckpt_path = experiment.get_checkpoint_path(target_model_id, epoch)

    ckpt_files = os.listdir(ckpt_path)
    if len(ckpt_files) < 4:
        return "SKIP", ckpt_path

    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print_with_tag("Successfully restored model: %d, epoch: %d" % (target_model_id, epoch),
                       print_tag, 1)
        return "SUCCESS", ckpt_path
    else:
        print_with_tag("No checkpoint file found", print_tag, 1)
        return "FAIL", ckpt_path


def make_saver(print_tag, experiment):
    variables_to_be_restored = \
        tf.trainable_variables() + tf.moving_average_variables() + tf.get_collection('TO_SAVE')
    print_with_tag(str(", ".join([v.name for v in variables_to_be_restored])), print_tag, 1)
    experiment.model_size.value = check_model_size(print_tag)
    return tf.train.Saver(variables_to_be_restored, max_to_keep=1)


def make_train_op(learning_rate, loss):
    # Optimizer operation is defined
    step_var = tf.Variable(0, trainable=False, name='step', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'TO_SAVE'])
    optimizer = tf.train.AdamOptimizer(learning_rate)

    trainable_vars = tf.trainable_variables()
    trainable_vars_wo_emb = filter(lambda v: 'embedding' not in v.name, trainable_vars)

    grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_vars)
    apply_gradients = optimizer.apply_gradients(grads_and_vars, global_step=step_var)

    grads_and_vars_wo_emb = optimizer.compute_gradients(loss, var_list=trainable_vars_wo_emb)
    grads_dict = {v.name: tf.reduce_mean(tf.abs(grads_and_vars_wo_emb[i][0]))
                  for i, v in enumerate(trainable_vars_wo_emb)}

    # Update moving averages with training
    ema_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    ema_updates_op = tf.group(*ema_updates)
    return tf.group(apply_gradients, ema_updates_op), grads_dict, trainable_vars_wo_emb, step_var


def output_loss(cross_entropy):
    regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    sum_of_reg_loss = tf.add_n(regularization_loss)
    return tf.add(cross_entropy, sum_of_reg_loss)


def output_cross_entropy(logits_op, onehot_gt):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_op, labels=onehot_gt))
    return cross_entropy


def prepare_y(input_gt, drop_whole_seq=False):
    if drop_whole_seq:
        # use whole dropped sequence.
        return all_drop(input_gt)
    else:
        # use random dropped sequence.
        return random_drop(input_gt)


def check_model_size(print_tag):
    model_size = 0
    size_dict = {}
    for tr_var in tf.trainable_variables():
        var_shape = tr_var.get_shape().as_list()
        temp_size = reduce(lambda x, y: x * y, var_shape)
        size_dict[tr_var.name] = temp_size
        model_size += reduce(lambda x, y: x * y, var_shape)

    print_with_tag("model size: %d" % model_size, print_tag, 1)
    for tr_var in tf.trainable_variables():
        print_with_tag("%s: %d" % (tr_var.name, size_dict[tr_var.name]), print_tag, 2)
    print_with_tag("", print_tag)
    return model_size
