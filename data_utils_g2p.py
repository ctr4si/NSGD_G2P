import codecs
import sys

import numpy as np
import numpy.random

from paths import *

SEPARATOR_GP = '  '
SEPARATOR_P = ' '
FILE_EXTENSION = '.dict'

ACCENT_SUFFIXES = {'0', '1', '2'}
MULTI_PRONUNCIATION_SUFFIXES = ['(1)', '(2)', '(3)']
MULTI_PRONUNCIATION_WORD_FILTERING_MODS = {'del', 'leave1st', 'leaveAll'}

# FIXME
MULTI_PRONUNCIATION_FILTERING = 'leaveAll'
# MULTI_PRONUNCIATION_FILTERING = 'leave1st'
# MULTI_PRONUNCIATION_FILTERING = 'del'
# ACCENT_REMOVAL = False
ACCENT_REMOVAL = True


def prepare_g2p_data(flags):
    print('\n\nInput files check.')

    if not os.path.exists(PROCESSED_DATA_PATH):
        os.mkdir(PROCESSED_DATA_PATH)
        do_preprocessing()

    if not (os.path.exists(INPUT_TR_PATH) and os.path.exists(INPUT_VA_PATH) and os.path.exists(INPUT_TE_PATH)):
        do_preprocessing()

    print('Processed files are ready.\n')
    sys.stdout.flush()

    train_lines = [l.strip() for l in codecs.open(INPUT_TR_PATH, 'r', 'utf-8').readlines()]
    valid_lines = [l.strip() for l in codecs.open(INPUT_VA_PATH, 'r', 'utf-8').readlines()]
    test_lines = [l.strip() for l in codecs.open(INPUT_TE_PATH, 'r', 'utf-8').readlines()]

    print('Vocabulary files check.')
    gr_vocab_file_path = GR_VOCAB_FILE_PATH
    ph_vocab_file_path = PH_VOCAB_FILE_PATH
    gr_base_vocabs = GR_START_VOCAB
    ph_base_vocabs = PH_START_VOCAB

    if os.path.exists(gr_vocab_file_path) and os.path.exists(ph_vocab_file_path):
        print('Loading vocabularies from %s\n' % PROCESSED_DATA_PATH)
        gr_vocab, gr_idx2char = load_vocabulary(gr_vocab_file_path)
        ph_vocab, ph_idx2char = load_vocabulary(ph_vocab_file_path)

    else:
        gr_vocab, gr_idx2char, ph_vocab, ph_idx2char = \
            create_vocabulary(train_lines + valid_lines + test_lines, gr_base_vocabs, ph_base_vocabs)

        save_vocabulary(gr_vocab, os.path.join(gr_vocab_file_path))
        save_vocabulary(ph_vocab, os.path.join(ph_vocab_file_path))
        print('Creating and saving vocab complete.\n')

    print('Generating sequence of ids from sequence of symbols...')
    sys.stdout.flush()
    train_gr_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[0].strip(), gr_vocab)
                    for line in train_lines]
    train_ph_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[1].strip().split(SEPARATOR_P), ph_vocab)
                    for line in train_lines]
    valid_gr_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[0].strip(), gr_vocab)
                    for line in valid_lines]
    valid_ph_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[1].strip().split(SEPARATOR_P), ph_vocab)
                    for line in valid_lines]
    test_gr_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[0].strip(), gr_vocab)
                   for line in test_lines]
    test_ph_ids = [symbols_to_ids(line.split(SEPARATOR_GP)[1].strip().split(SEPARATOR_P), ph_vocab)
                   for line in test_lines]

    train_sources, train_targets = np.array(train_gr_ids), np.array(train_ph_ids)
    valid_sources, valid_targets = np.array(valid_gr_ids), np.array(valid_ph_ids)
    test_sources, test_targets = np.array(test_gr_ids), np.array(test_ph_ids)

    print("All's ready.\n")
    sys.stdout.flush()

    return train_sources, train_targets, valid_sources, valid_targets, test_sources, test_targets, \
           gr_vocab, gr_idx2char, ph_vocab, ph_idx2char


def do_preprocessing():
    print('There is no preprocessed files.')
    print('Now, preprocessing...')
    sys.stdout.flush()

    processed_file_path = RAW_INPUT_FILE_PATH.split(FILE_EXTENSION)[0]
    processed_file_path += ("_mul-%s_%sAcc"
                            % (MULTI_PRONUNCIATION_FILTERING, 'no' if ACCENT_REMOVAL else 'with')) + FILE_EXTENSION

    words_prons_dict = preprocessing_raw_dataset(RAW_INPUT_FILE_PATH,
                                                 multi_pronunciation_filtering=MULTI_PRONUNCIATION_FILTERING,
                                                 remove_acc=ACCENT_REMOVAL)

    raw_dataset_partition(words_prons_dict, processed_file_path)


def preprocessing_raw_dataset(file_path, multi_pronunciation_filtering='del', remove_acc=False):
    assert multi_pronunciation_filtering in MULTI_PRONUNCIATION_WORD_FILTERING_MODS
    f = open(file_path, 'r')
    lines = filter(lambda x: ';;;' not in x[:3], f.readlines())
    lines = filter(lambda x: is_ascii(x), lines)  # delete non-ascii words

    word_pron_dict = {}
    for l in lines:
        word, pron = l.split(SEPARATOR_GP)
        word, pron = word.strip(), pron.strip()

        if word[-3:] in MULTI_PRONUNCIATION_SUFFIXES:
            word = word[:-3]

        phonemes = pron.split(SEPARATOR_P)

        if remove_acc:
            phonemes = [ph[:-1] if ph[-1] in ACCENT_SUFFIXES else ph for ph in phonemes]

        if word in word_pron_dict:
            word_pron_dict[word].append(phonemes)
        else:
            word_pron_dict[word] = [phonemes]

    multi_pron_words = [(w, len(p_list)) for w, p_list in word_pron_dict.items()]
    multi_pron_words = filter(lambda x: x[1] > 1, multi_pron_words)

    for multi_pron_word in multi_pron_words:
        if multi_pronunciation_filtering == 'del':
            # delete words having multiple pronunciations
            del word_pron_dict[multi_pron_word[0]]

        elif multi_pronunciation_filtering == 'leave1st':
            word_pron_dict[multi_pron_word[0]] = [word_pron_dict[multi_pron_word[0]][0]]

        else:
            break

    f.close()
    return word_pron_dict


def raw_dataset_partition(words_pron_dict, file_path):
    base_name = file_path.split(FILE_EXTENSION)[0] + "%s" + FILE_EXTENSION
    f_tr = open(base_name % '_tr', 'w')
    f_va = open(base_name % '_va', 'w')
    f_te = open(base_name % '_te', 'w')

    num_words = len(words_pron_dict)
    num_words_te = int(num_words * 0.1)
    num_words_va = int(num_words * 0.05)

    words_list = sorted(words_pron_dict.keys())
    index = np.arange(num_words)

    numpy.random.seed(1)
    numpy.random.shuffle(index)

    index_te = [(i, 'te') for i in index[:num_words_te]]
    index_va = [(i, 'va') for i in index[num_words_te:][:num_words_va]]
    index_tr = [(i, 'tr') for i in index[num_words_te:][num_words_va:]]
    index = sorted(index_te + index_va + index_tr)

    for i, w in enumerate(words_list):
        phonemes_list = words_pron_dict[w]
        strings = [w + SEPARATOR_GP + SEPARATOR_P.join(phonemes) + '\n' for phonemes in phonemes_list]

        if index[i][1] == 'te':
            writer = f_te
        elif index[i][1] == 'va':
            writer = f_va
        else:
            writer = f_tr

        for str_to_write in strings:
            writer.write(str_to_write)

    f_tr.close()
    f_va.close()
    f_te.close()


def create_vocabulary(data, gr_base_vocabs, ph_base_vocabs):
    gr_list, ph_list = gr_base_vocabs[:], ph_base_vocabs[:]

    for line in data:
        for item in line.split(SEPARATOR_GP)[0].strip():
            if item not in gr_list:
                gr_list.append(item)
        for item in line.split(SEPARATOR_GP)[1].split(SEPARATOR_P):
            if item not in ph_list:
                ph_list.append(item)
    gr_vocab = dict([(x, y) for (y, x) in enumerate(gr_list)])
    ph_vocab = dict([(x, y) for (y, x) in enumerate(ph_list)])

    gr_idx2char = dict([(y, x) for (y, x) in enumerate(gr_list)])
    ph_idx2char = dict([(y, x) for (y, x) in enumerate(ph_list)])
    return gr_vocab, gr_idx2char, ph_vocab, ph_idx2char


def save_vocabulary(vocab, vocabulary_path):
    print('Creating vocabulary %s' % vocabulary_path)
    with codecs.open(vocabulary_path, 'w', 'utf-8') as vocab_file:
        for symbol in sorted(vocab, key=vocab.get):
            vocab_file.write(symbol + '\n')


def load_vocabulary(vocabulary_path):
    with codecs.open(vocabulary_path, 'r', 'utf-8') as vocab_file:
        vocab = [line.strip() for line in vocab_file.readlines()]
    return dict([(x, y) for (y, x) in enumerate(vocab)]), dict([(y, x) for (y, x) in enumerate(vocab)])


def symbols_to_ids(symbols, vocab):
    ids = []
    max_len = MAXLEN

    for s in list(symbols):
        if s not in vocab:
            print "%s not in vocab!" % s
            sys.exit(-1)
        ids.append(vocab.get(s))

    ids += [vocab.get(PAD)] * (max_len - len(ids))
    return ids


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


if __name__ == "__main__":
    do_preprocessing()
