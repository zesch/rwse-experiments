from transformers import AutoTokenizer

import pandas as pd


def extract_word_statistics(file_name, nlp, list_of_words):
    """

    :param file_name: input file data expected to be from the leipzig corpora (indexed sentences)
    :param nlp: model used for spacy tokenization
    :param list_of_words: words to look for in each tokenized sentence
    :return: DataFrame with list of sentence indices by word form list_of_words
    """
    word_statistics = {word: [] for word in list_of_words}
    with open(file_name, 'r') as f:
        for line in f.readlines():
            idx, sentence = line.split('\t')
            doc = nlp(sentence)
            for token in doc:
                for word in list_of_words:
                    if word == token.text:
                        word_statistics[word].append(idx)
    for key, value in word_statistics.items():
        word_statistics[key] = ','.join(value)
    temp_df = pd.DataFrame(word_statistics.items(), columns=('word', 'sentence_indices'))
    return temp_df


def collect_confusion_set_frequencies(input_df: pd.DataFrame, confusion_set_strings):
    word_frequencies = dict(zip(input_df['word'], input_df['frequency']))

    confusion_set_frequencies = dict()

    for confusion_set in confusion_set_strings:
        confusion_set_frequencies[confusion_set] = 0
        confusion_set_list = confusion_set.split(',')
        for word, frequency in word_frequencies.items():
            if word in confusion_set_list:
                confusion_set_frequencies[confusion_set] += frequency

    keys = confusion_set_frequencies.keys()
    values = confusion_set_frequencies.values()

    return pd.DataFrame({'confusion_set': keys, 'frequency': values}, index=keys)


def read_classification_report(file_name:str):
    result = dict()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            confusion_set, num_matches, num_sequences = line.strip().split(';')
            result[confusion_set] = (int(num_matches), int(num_sequences))
    return result


def normalize_report(file_name:str):
    report_data = read_classification_report(file_name)
    normalized_report = dict()
    for key, item in report_data.items():
        if item[1] > 0:
            value = item[0] / item[1]
            normalized_report[key] = value
        else:
            normalized_report[key] = 0
    return normalized_report


def check_token(tokenizer:AutoTokenizer, target:str) ->bool:
    target_enc = tokenizer.tokenize(target)
    if len(target_enc) > 1 or target_enc[0] == tokenizer.unk_token:
        return None
    return target


def read_classification_report(file_name:str):
    result = dict()
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            confusion_set, num_matches, num_sequences = line.strip().split(';')
            result[confusion_set] = (int(num_matches), int(num_sequences))
    return result


def calculate_false_alarm_rate(file_name):
    data = read_classification_report(file_name)
    FP = sum(value[0] for value in data.values())
    TN = sum(value[1] for value in data.values()) - FP
    return FP/(TN+FP)


def calculate_miss_rate(file_name):
    data = read_classification_report(file_name)
    TP = sum(value[0] for value in data.values())
    FN = sum(value[1] for value in data.values()) - TP
    return FN/(TP+FN)


def calculate_accuracy(false_positives, true_positives):
    data = read_classification_report(false_positives)
    FP = sum(value[0] for value in data.values())
    TN = sum(value[1] for value in data.values()) - FP
    data = read_classification_report(true_positives)
    TP = sum(value[0] for value in data.values())
    FN = sum(value[1] for value in data.values()) - TP
    return (TP+TN)/(TN+FP+TP+FN)