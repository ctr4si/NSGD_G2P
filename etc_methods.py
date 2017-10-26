__author__ = 'jjamjung'

import sys

import editdistance as ed
import numpy as np
import pandas as pd


def random_drop(input_batch):
    shape = input_batch.shape

    rand = np.random.rand()
    rand_pow = rand ** 2
    rand_threshold = rand_pow

    random_batch = np.random.rand(shape[0], shape[1])
    mul_batch = np.zeros(shape)
    mul_batch[random_batch < rand_threshold] = 1
    new_input_batch = np.multiply(mul_batch, input_batch)

    return new_input_batch


def all_drop(input_batch):
    return np.zeros_like(input_batch)


def get_batch(x, y, curr_index, batch_size):
    batch_x = x[curr_index:curr_index + batch_size, :]
    batch_y = y[curr_index:curr_index + batch_size, :]
    return batch_x, batch_y


def get_instances_real_length(input_array, id_to_ignore):
    indices_to_ignore = np.where(input_array == id_to_ignore)
    indices_to_ignore = np.stack([indices_to_ignore[0], indices_to_ignore[1]], axis=1)
    indices_to_ignore = pd.DataFrame(indices_to_ignore, columns=['row', 'col'])

    first_index_to_ignore_each_row = indices_to_ignore.groupby('row').first()
    first_index_to_ignore_each_row = first_index_to_ignore_each_row.reset_index()
    first_index_to_ignore_each_row = first_index_to_ignore_each_row.values

    # eos_index: first padding index list (and also meaning each word length)
    row_index, eos_index = first_index_to_ignore_each_row[:, 0].tolist(), first_index_to_ignore_each_row[:, 1].tolist()
    pair_list = zip(row_index, eos_index)
    max_len = len(input_array[0])

    if len(eos_index) < len(input_array):
        for i in range(len(input_array)):
            if i not in row_index:
                pair_list.append((i, max_len))
        pair_list = sorted(pair_list)
        eos_index = [p[1] for p in pair_list]

    assert len(eos_index) == len(input_array)

    return eos_index


def compute_batch_edit_distances_for_decoding(source_arr, prediction_arr, ground_truth_arr, experiment):
    batch_size = len(ground_truth_arr)

    source_lengths = get_instances_real_length(source_arr, id_to_ignore=experiment.get_source_ending_id())
    pred_lengths = get_instances_real_length(prediction_arr, id_to_ignore=experiment.get_target_ending_id())
    gt_lengths = get_instances_real_length(ground_truth_arr, id_to_ignore=experiment.get_target_ending_id())

    edit_distances = []
    for i in range(batch_size):
        s_seq = source_arr[i][:source_lengths[i]]
        p_seq = prediction_arr[i][:pred_lengths[i]]
        g_seq = ground_truth_arr[i][:gt_lengths[i]]

        distance = ed.eval(g_seq, p_seq)
        edit_distances.append((list(s_seq), distance, len(g_seq)))

    return edit_distances


def dataset_shuffling(x, y):
    shuffled_idx = np.arange(len(y))
    np.random.shuffle(shuffled_idx)
    return x[shuffled_idx, :], y[shuffled_idx, :]


def ids_to_symbols(ids, vocab):
    symbols = []
    if len(ids.shape) == 2:
        for i in range(ids.shape[0]):
            symbols_i = []
            for j in range(ids.shape[1]):
                symbols_i.append(vocab[ids[i][j]])
            symbols.append(symbols_i)
    else:
        for i in range(ids.shape[0]):
            symbols_i = []
            for j in range(ids.shape[1]):
                symbols_ij = []
                for k in range(ids.shape[2]):
                    symbols_ij.append(vocab[ids[i][j][k]])
                symbols_i.append(symbols_ij)
            symbols.append(symbols_i)
    return symbols


def ids_to_strings(ids):
    strings = []
    if len(ids.shape) == 2:
        for i in range(ids.shape[0]):
            strings_i = []
            for j in range(ids.shape[1]):
                strings_i.append(str(ids[i][j]))
            strings.append(strings_i)
    else:
        for i in range(ids.shape[0]):
            strings_i = []
            for j in range(ids.shape[1]):
                strings_ij = []
                for k in range(ids.shape[2]):
                    strings_ij.append(str(ids[i][j][k]))
                strings_i.append(strings_ij)
            strings.append(strings_i)
    return strings


def eval_result_to_string(results, tag):
    if type(results) == np.ndarray:
        performance_measures = ['PER', 'WER', 'loss']
    else:
        performance_measures = ['PER', 'WER']

    performance_measures = [tag + '_' + pm for pm in performance_measures]
    performance_measures = [pm + ': %.4f' for pm in performance_measures]

    performances = list(results)
    performances[0] *= 100
    performances[1] *= 100
    performances = tuple(performances)

    result_string = ', '.join(performance_measures)
    result_string %= performances
    return result_string


def print_with_tag(content_to_print, tag_to_print, indent_level=0):
    print tag_to_print + ("  " * indent_level) + str(content_to_print)
    sys.stdout.flush()
