import csv
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
    list_of_words = sorted(set(item.lower() for item in list_of_words))
    word_statistics = {word: 0 for word in list_of_words}
    with open(file_name, 'r') as f:
        sentences = f.readlines()
        for sentence in sentences:
            doc = nlp(sentence)
            for token in doc:
                for word in list_of_words:
                    if word == token.text.lower():
                        word_statistics[word] += 1
    temp_df = pd.DataFrame(word_statistics.items(), columns=('word', 'frequency'))
    return temp_df


def transform_wortschatz_leipzig(file_name):
    with open(file_name, 'r') as f:
        sentences = f.readlines()
    # sentences of Wortschatz Leipzig are enumerated -> remove enumeration
    sentences_cleaned = [sentence.split('\t')[1].strip() for sentence in sentences]
    with open(f'{file_name}_transformed', 'w') as f:
        for sentence in sentences_cleaned:
            print(sentence, file=f, end='\n')


def load_confusion_sets_from_file(file_path):
    result_sets = {}
    with open(file_path, 'r') as file:
        for row in csv.reader(file):
            for key in row:
                result_sets[key] = list(row)
    return result_sets
