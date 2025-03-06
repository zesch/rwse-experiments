from transformers import AutoTokenizer

import numpy as np
import pandas as pd

def collect_sentences_by_word(list_of_words, list_of_sentences, nlp):
    sentences_by_word = {}
    for word in list_of_words:
        sentences_by_word[word] = []
    for sentence in list_of_sentences:
        doc = nlp(sentence)
        for token in doc:
            for word in list_of_words:
                if word == token.text:
                    sentences_by_word[word].append(sentence)
    for key, value in sentences_by_word.items():
        sentences_by_word[key] = list(set(value))
    return sentences_by_word


def collect_sentences_by_confusion_sets(list_of_confusion_sets, list_of_sentences, nlp):
    words = set()
    sentences_by_confusion_set = {}
    for confusion_set in list_of_confusion_sets:
        words = words.union(confusion_set)
    sentences_by_word = collect_sentences_by_word(list(words), list_of_sentences, nlp)
    for confusion_set in list_of_confusion_sets:
        key = ','.join(confusion_set)
        sentences_by_confusion_set[key] = list()
        for word in words:
            if word in confusion_set:
                sentences_by_confusion_set[key].extend(sentences_by_word[word])
    for key, value in sentences_by_confusion_set.items():
        sentences_by_confusion_set[key] = list(set(value))
    return sentences_by_confusion_set


def replace_confusion_set_words_in_sentences(list_of_sentences_by_confusion_sets, nlp):
    result = dict()
    for confusion_set, sentences in list_of_sentences_by_confusion_sets.items():
        tmp_sentences = []
        list_of_words = confusion_set.split(',')
        for word in list_of_words:
            for partner in list_of_words:
                if word.lower() != partner.lower(): # ignore case
                    for sentence in sentences:
                        doc = nlp(sentence)
                        for token in doc:
                            if token.text == word:
                                tmp_dict = {
                                    'org_word':token.text,
                                    'new_token':{
                                        'word':partner,
                                        'begin':token.idx,
                                        'end':token.idx+len(partner),
                                    },
                                    'sentence':sentence[:token.idx] + partner + sentence[token.idx+len(token.text):]
                                }
                                tmp_sentences.append(tmp_dict)
        result[confusion_set] = tmp_sentences
    return result


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


def calculate_metrics(file_false_positives:str, file_true_positives:str):
    confusion_matrix, report_data = calculate_confusion_matrix(file_false_positives, file_true_positives)

    data_mean_sensitivity = normalize_report(file_true_positives)
    data_mean_fpr = normalize_report(file_false_positives)

    metrics = dict()
    metrics['precision'] = report_data['TP'] / (report_data['TP'] + report_data['FP'])
    metrics['mean false-positive rate'] = sum(data_mean_fpr.values()) / len(data_mean_fpr)
    metrics['mean sensitivity (mean recall)'] = sum(data_mean_sensitivity.values()) / len(data_mean_sensitivity)
    metrics['sensitivity (recall)'] = report_data['TP'] / (report_data['FN'] + report_data['TP'])
    metrics['specificity'] = report_data['TN'] / (report_data['FP'] + report_data['TN'])
    metrics['f1'] = 2 * metrics['precision'] * metrics['sensitivity (recall)'] / (metrics['precision'] + metrics['sensitivity (recall)'])
    return metrics, confusion_matrix


def calculate_confusion_matrix(file_false_positives:str, file_true_positives:str):
    confusion_matrix_dict = dict()
    data_false_positives = read_classification_report(file_false_positives)

    confusion_matrix_dict['FP'] = sum(value[0] for value in data_false_positives.values())
    confusion_matrix_dict['TN'] = sum(value[1] for value in data_false_positives.values()) - confusion_matrix_dict['FP']

    data_true_positives = read_classification_report(file_true_positives)
    confusion_matrix_dict['TP'] = sum(value[0] for value in data_true_positives.values())
    confusion_matrix_dict['FN'] = sum(value[1] for value in data_true_positives.values()) - confusion_matrix_dict['TP']

    confusion_matrix = np.matrix([
        [confusion_matrix_dict['TP'], confusion_matrix_dict['FN']],
        [confusion_matrix_dict['FP'], confusion_matrix_dict['TN']]
    ])

    return confusion_matrix, confusion_matrix_dict


def normalize_report(file_name:str):
    report_data = read_classification_report(file_name)
    normalized_report = dict()
    for key, item in report_data.items():
        if item[1] > 0:
            value = item[0] / item[1]
            #print(f'{key}: {value:.10f}')
            normalized_report[key] = value
        else:
            normalized_report[key] = 0
    return normalized_report


def check_token(tokenizer:AutoTokenizer, target:str) ->bool:
    target_enc = tokenizer.tokenize(target)
    if len(target_enc) > 1 or target_enc[0] == tokenizer.unk_token:
        return None
    return target