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
