__author__ = 'jjamjung'

import sys

import pandas as pd
from datetime import datetime
from pytz import timezone

from etc_methods import print_with_tag
from paths import *


class Parameter(object):
    def __init__(self, name, value, db_key=False, pretrain_key=False):
        self.name = name
        self.value = value
        self.db_key = db_key
        self.pretrain_key = pretrain_key

    def __str__(self):
        return "%s: %s" % (self.name, self.value)


class Experiment(object):
    """Class with training parameters."""

    def __init__(self, source_vocab, source_idx2char, target_vocab, target_idx2char, flags, version, print_tag='[DB]'):
        self.print_tag = print_tag

        self.target_vocab = Parameter('target_vocab', target_vocab)
        self.target_idx2char = Parameter('target_idx2char', target_idx2char)
        self.source_vocab = Parameter('source_vocab', source_vocab)
        self.source_idx2char = Parameter('source_idx2char', source_idx2char)

        self.num_source_vocabs = Parameter('num_source_vocabs', len(source_vocab))
        self.num_target_vocabs = Parameter('num_target_vocabs', len(target_vocab))

        """ Experiment info objects"""
        # used code version and data set
        self.version = Parameter('version', version, db_key=True, pretrain_key=True)
        self.input_file_keyword = Parameter('input_files_keyword', INPUT_FILES_KEYWORD,
                                            db_key=True, pretrain_key=True)

        self.static_info_list = [self.version, self.input_file_keyword]

        # ids are re-assigned after searching DB
        self.model_id = Parameter('model_id', 1)  # should be a unique index
        self.exp_id = Parameter('exp_id', 1)

        self.model_id_info_params = [self.model_id, self.exp_id]

        # Start and end time of training
        self.start_time = Parameter('start_time', str(datetime.now(timezone('Asia/Seoul'))))
        self.end_time = Parameter('end_time', None)

        self.time_info_list = [self.start_time, self.end_time]

        """ Result objects"""
        self.best_epoch = Parameter('best_epoch', None)
        self.va_per = Parameter('va_per', None)
        self.va_wer = Parameter('va_wer', None)
        self.te_per = Parameter('te_per', None)
        self.te_wer = Parameter('te_wer', None)

        self.results_list = [self.best_epoch,
                             self.va_per,
                             self.va_wer,
                             self.te_per,
                             self.te_wer]

        """ Running info objects"""
        self.gpu_memory_fraction = Parameter('gpu_memory_fraction', flags.gpu_memory_fraction)
        self.model_parameter_saving = Parameter('model_parameter_saving', flags.model_parameter_saving)
        self.eval_only = Parameter('eval_only', flags.eval_only)

        # Only if 'load_pre_trained_model' is True or 'eval_only' is True, this argument is used.
        self.model_id_to_load = \
            Parameter('model_id_to_load',
                      flags.model_id_to_load if flags.eval_only else -1,
                      db_key=False)

        self.running_info_params = [self.gpu_memory_fraction,
                                    self.model_parameter_saving,
                                    self.eval_only,
                                    self.model_id_to_load]

        """ Model architecture """
        self.max_len = Parameter('max_len', MAXLEN, db_key=True, pretrain_key=True)
        self.embedding_size = Parameter('embedding_size', flags.embedding_size, db_key=True, pretrain_key=True)

        # conv structures using source sequence
        self.source_num_layers = Parameter('source_num_layers', flags.source_num_layers, db_key=True, pretrain_key=True)
        self.source_filter_width = Parameter('source_filter_width', flags.source_filter_width,
                                             db_key=True, pretrain_key=True)

        # conv structures using target sequence
        self.target_num_layers = Parameter('target_num_layers', flags.target_num_layers, db_key=True, pretrain_key=True)
        self.target_filter_width = Parameter('target_filter_width', flags.target_filter_width,
                                             db_key=True, pretrain_key=True)

        # conv structures for decoding
        self.decoding_num_layers = Parameter('decoding_num_layers', flags.decoding_num_layers,
                                             db_key=True, pretrain_key=True)
        self.decoding_filter_width = Parameter('decoding_filter_width', flags.decoding_filter_width,
                                               db_key=True, pretrain_key=True)

        self.model_size = Parameter('model_size', None)

        self.model_architecture_params = [self.embedding_size,
                                          self.source_num_layers,
                                          self.source_filter_width,
                                          self.target_num_layers,
                                          self.target_filter_width,
                                          self.decoding_num_layers,
                                          self.decoding_filter_width,
                                          self.model_size]

        """ Learning algorithm """
        self.learning_rate = Parameter('learning_rate', flags.learning_rate, db_key=True)
        self.learning_rate_decay_factor = Parameter('learning_rate_decay_factor',
                                                    flags.learning_rate_decay_factor, db_key=True)
        self.adapting_cycle_steps = Parameter('adapting_cycle_steps', flags.adapting_cycle_steps, db_key=True)
        self.adapting_queue_size = Parameter('adapting_queue_size', flags.adapting_queue_size, db_key=True)
        self.batch_size = Parameter('batch_size', flags.batch_size, db_key=True)
        self.num_epochs = Parameter('num_epochs', flags.num_epochs, db_key=True)

        self.learning_algorithm_params = [self.learning_rate,
                                          self.learning_rate_decay_factor,
                                          self.adapting_cycle_steps,
                                          self.adapting_queue_size,
                                          self.batch_size,
                                          self.num_epochs]

        """ etc """
        # Debugging
        self.inference_debugging = Parameter('inference_debugging', flags.inference_debugging)

        # Only if 'inference_debugging' is True, this argument is used.
        self.wrong_only_debugging = \
            Parameter('wrong_only_debugging', flags.wrong_only_debugging if flags.inference_debugging else False)

        # Only if 'eval_only' is False, this argument is used.
        self.summary_write = Parameter('summary_write', False if flags.eval_only else flags.summary_write)
        self.summary_step_cycle = Parameter('summary_step_cycle', 0 if flags.eval_only else flags.summary_step_cycle)

        # Regularization
        self.weight_decay = Parameter('weight_decay', flags.weight_decay, db_key=True, pretrain_key=True)

        self.residual_reg_keep_prob = \
            Parameter('residual_reg_keep_prob',
                      flags.residual_reg_keep_prob,
                      db_key=True,
                      pretrain_key=True)

        self.etc_parameters = [self.inference_debugging,
                               self.wrong_only_debugging,
                               self.summary_write,
                               self.summary_step_cycle,
                               self.weight_decay,
                               self.residual_reg_keep_prob]

        self.all_params = self.static_info_list + self.model_id_info_params
        self.all_params += self.time_info_list + self.running_info_params
        self.all_params += self.model_architecture_params + self.learning_algorithm_params
        self.all_params += self.etc_parameters + self.results_list

        self.db_path = MODEL_DB_PATH
        self.summary_base_path = SUMMARY_BASE_PATH
        self.checkpoint_base_path = CHECKPOINT_BASE_PATH

        self.db_keys = filter(lambda k: k.db_key, self.all_params)
        self.pretrain_keys = filter(lambda k: k.pretrain_key, self.all_params)

        self.db_table = None
        self.read_db_table()
        self.feasibility_check()

        if flags.eval_only:
            print_with_tag("Read experiment info from DB using given model id, %d" % self.model_id_to_load.value,
                           self.print_tag, 1)
            self.update_parameters(self.search_db(self.db_keys + [Parameter(self.model_id.name,
                                                                            self.model_id_to_load.value)]))
        else:
            self.search_and_assign_new_index(self.db_keys)

    def read_db_table(self, do_print=True):
        if os.path.exists(self.db_path):
            self.db_table = pd.read_csv(self.db_path)
            if do_print:
                print_with_tag("DB table is loaded: %s" % self.db_path, self.print_tag, 1)
        else:
            if do_print:
                print_with_tag("DB table does not exist", self.print_tag, 1)

    def make_db_row_dataframe(self):
        row = [p.value for p in self.all_params]
        cols = [p.name for p in self.all_params]
        return pd.DataFrame([row], columns=cols)

    def search_db(self, keys):
        if self.db_table is None:
            return None

        assert len(keys) > 0

        booleans = []
        for key in keys:
            booleans.append(self.db_table[key.name] == key.value)

        if len(keys) > 2:
            search_result = booleans[0] & booleans[1]
            for i in range(len(booleans))[2:]:
                search_result = search_result & booleans[i]
            return self.db_table.loc[search_result]
        elif len(keys) == 1:
            search_result = booleans[0]
            return self.db_table.loc[search_result]
        else:
            return None

    def search_and_assign_new_index(self, keys):
        search_result = self.search_db(keys)
        if search_result is not None:
            if len(search_result) > 0:
                self.exp_id.value = search_result[self.exp_id.name].max()
            else:
                self.exp_id.value = self.get_new_exp_idx()

        self.model_id.value = self.get_new_model_idx()

        print_with_tag("Read DB and assign new experiment ids, model id: %d, exp id: %d"
                       % (self.model_id.value, self.exp_id.value),
                       self.print_tag, 1)
        self.update_db_table()

    def get_new_model_idx(self):
        if self.db_table is None:
            return 1
        else:
            return self.db_table[self.model_id.name].max() + 1

    def get_new_exp_idx(self):
        if self.db_table is None:
            return 1
        else:
            return self.db_table[self.exp_id.name].max() + 1

    def update_parameters(self, db_row_df):
        db_row_series = db_row_df.iloc[0]
        for p in self.all_params:
            p.value = db_row_series[p.name]

    def update_va_performances(self, per, wer, epoch):
        self.va_per.value = per
        self.va_wer.value = wer
        self.best_epoch.value = epoch

    def update_te_performances(self, per, wer):
        self.te_per.value = per
        self.te_wer.value = wer

    def update_db_table(self):
        self.read_db_table(do_print=False)
        db_row_df = self.make_db_row_dataframe()

        if self.db_table is None:
            self.db_table = db_row_df
        else:
            bool_index = self.db_table[self.model_id.name] == self.model_id.value

            assert len(self.db_table.loc[bool_index]) < 2

            if bool_index.any():
                self.db_table.loc[bool_index] = db_row_df.values
            else:
                self.db_table = pd.concat([self.db_table, db_row_df], axis=0)
        self.db_table.to_csv(self.db_path, index=False)
        print_with_tag("DB table is updated. (model id: %d)" % self.model_id.value, self.print_tag, 1)

    def finish_training(self):
        self.end_time.value = str(datetime.now(timezone('Asia/Seoul')))
        self.update_db_table()

    def finish_test(self):
        self.update_db_table()

    def initialize_directories(self):
        checkpoint_dir = os.path.join(self.checkpoint_base_path, str(self.model_id.value))
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        summary_dir = os.path.join(self.summary_base_path, str(self.model_id.value))
        if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)

        return summary_dir

    def get_target_pad_id(self):
        return self.target_vocab.value.get(PAD)

    def get_source_pad_id(self):
        return self.source_vocab.value.get(PAD)

    def get_target_blk_id(self):
        return self.target_vocab.value.get(BLK)

    def get_target_ending_id(self):
        return self.get_target_pad_id()

    def get_source_ending_id(self):
        return self.get_source_pad_id()

    def get_total_step_num(self, dataset_size, batch_size):
        return self.num_epochs.value * dataset_size // batch_size

    def get_epoch_in_float(self, step, dataset_size):
        return float(step) * self.batch_size.value / dataset_size

    def get_curr_epoch_in_float(self, step, dataset_size):
        return self.get_epoch_in_float(step + 1, dataset_size)

    def get_prev_epoch_in_float(self, step, dataset_size):
        return self.get_epoch_in_float(step, dataset_size)

    def get_curr_epoch_in_int(self, step, dataset_size):
        epoch_float = self.get_curr_epoch_in_float(step, dataset_size)
        return int(round(epoch_float))

    def time_to_check_loss(self, step):
        return step % self.adapting_cycle_steps.value == 0 and step > 0

    def time_to_save_model(self, step, total_step, dataset_size):
        epoch_float = self.get_curr_epoch_in_float(step, dataset_size)
        epoch_int = self.get_curr_epoch_in_int(step, dataset_size)
        prev_epoch_float = self.get_prev_epoch_in_float(step, dataset_size)

        cond = epoch_float >= epoch_int > prev_epoch_float > 0  # new epoch
        cond |= total_step - step == 1  # the last step
        return cond

    def time_to_summarize(self, step):
        return self.summary_write.value and step % self.summary_step_cycle.value == 0 and step > 0

    def get_checkpoint_path(self, model_idx, epoch):
        ckpt_path = os.path.join(self.checkpoint_base_path, str(model_idx))
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        ckpt_path = os.path.join(ckpt_path, str(epoch))
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

        return ckpt_path

    def feasibility_check(self):
        if self.db_table is None:
            if self.eval_only.value:
                print_with_tag("ERROR: You asked load trained model, but DB table does not exist..", self.print_tag, 1)
                sys.exit(-1)

        else:
            if self.eval_only.value:
                search_keys = self.db_keys + [self.model_id_to_load]
            else:
                return

            search_result = self.search_db(search_keys)
            if search_result is None:
                print_with_tag("ERROR: there is no model with this setting..\n", self.print_tag, 1)
                for key in search_keys:
                    print_with_tag(str(key), self.print_tag, 2)
                sys.exit(-1)

        print_with_tag("Feasibility check, passed", self.print_tag, 1)

    def print_performances(self):
        print_with_tag("Model: %d, Epoch: %d, VA_PER: %.3f%%, VA_WER: %.3f%%, TE_PER: %.3f%%, TE_WER: %.3f%%" %
                       (self.model_id.value, self.best_epoch.value, self.va_per.value * 100, self.va_wer.value * 100,
                        self.te_per.value * 100, self.te_wer.value * 100),
                       self.print_tag, 2)

    def print_all_params(self):
        print_with_tag("", self.print_tag)
        print_with_tag("Parameters for GREEDY INFERENCE", self.print_tag, 1)
        for db_key in self.all_params:
            print_with_tag(str(db_key), self.print_tag, 2)
        print_with_tag("", self.print_tag)
