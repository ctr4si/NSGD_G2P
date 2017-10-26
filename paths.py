# -*- coding: utf-8 -*-
__author__ = 'jjamjung'

import os

VERSION = '1.5.0'

""" data """
DATA_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
if not os.path.exists(DATA_BASE_PATH):
    os.mkdir(DATA_BASE_PATH)

RAW_INPUT_FILE_NAME = 'cmudict-0.7b.dict'
RAW_INPUT_FILE_PATH = os.path.join(DATA_BASE_PATH, RAW_INPUT_FILE_NAME)

INPUT_FILE_EXTENSION = '.dict'

# FIXME
INPUT_FILES_KEYWORD = RAW_INPUT_FILE_NAME.split(INPUT_FILE_EXTENSION)[0] + '_mul-leaveAll_noAcc'
# INPUT_FILES_KEYWORD = RAW_INPUT_FILE_NAME.split(INPUT_FILE_EXTENSION)[0] + '_mul-leaveAll_withAcc'
# INPUT_FILES_KEYWORD = RAW_INPUT_FILE_NAME.split(INPUT_FILE_EXTENSION)[0] + '_mul-leave1st_withAcc'
# INPUT_FILES_KEYWORD = RAW_INPUT_FILE_NAME.split(INPUT_FILE_EXTENSION)[0] + '_mul-del_withAcc'

INPUT_TR_PATH = os.path.join(DATA_BASE_PATH, INPUT_FILES_KEYWORD + "_tr" + INPUT_FILE_EXTENSION)
INPUT_VA_PATH = os.path.join(DATA_BASE_PATH, INPUT_FILES_KEYWORD + "_va" + INPUT_FILE_EXTENSION)
INPUT_TE_PATH = os.path.join(DATA_BASE_PATH, INPUT_FILES_KEYWORD + "_te" + INPUT_FILE_EXTENSION)

PROCESSED_DATA_PATH = os.path.join(DATA_BASE_PATH, INPUT_FILES_KEYWORD)
if not os.path.exists(PROCESSED_DATA_PATH):
    os.mkdir(PROCESSED_DATA_PATH)

""" model """
MODEL_BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
if not os.path.exists(MODEL_BASE_PATH):
    os.mkdir(MODEL_BASE_PATH)

SUMMARY_BASE_PATH = os.path.join(MODEL_BASE_PATH, INPUT_FILES_KEYWORD + '_summary-' + VERSION)
if not os.path.exists(SUMMARY_BASE_PATH):
    os.mkdir(SUMMARY_BASE_PATH)

CHECKPOINT_BASE_PATH = os.path.join(MODEL_BASE_PATH, INPUT_FILES_KEYWORD + '_checkpoint-' + VERSION)
if not os.path.exists(CHECKPOINT_BASE_PATH):
    os.mkdir(CHECKPOINT_BASE_PATH)

# ...g2p/model/keyword_DB-version/
MODEL_DB_PATH = os.path.join(MODEL_BASE_PATH, INPUT_FILES_KEYWORD + "_DB-" + VERSION + ".csv")

""" preprocessing """
UNK = '_UNK'
PAD = '_PAD'
BLK = '_BLK'
EOS = '_EOS'
SOS = '_SOS'
GR_START_VOCAB, PH_START_VOCAB = [PAD], [BLK, PAD]

GR_VOCAB_FILE_PATH = os.path.join(PROCESSED_DATA_PATH, 'vocab.grapheme')
PH_VOCAB_FILE_PATH = os.path.join(PROCESSED_DATA_PATH, 'vocab.phoneme')

MAXLEN = 35
