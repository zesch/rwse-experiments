import spacy


def collect_sentences_by_word(list_of_words, list_of_sentences):
    nlp = spacy.load("en_core_web_sm")
    sentences_by_word = {}
    for word in list_of_words:
        sentences_by_word[word] = set()
    for sentence in list_of_sentences:
        doc = nlp(sentence)
        for token in doc:
            for word in list_of_words:
                if word == token.text:
                    sentences_by_word[word] = sentences_by_word[word].union((sentence,))
    return sentences_by_word

def collect_sentences_by_confusion_sets(list_of_confusion_sets, list_of_sentences):
    words = set()
    sentences_by_confusion_set = {}
    for confusion_set in list_of_confusion_sets:
        words = words.union(confusion_set)
    sentences_by_word = collect_sentences_by_word(list(words), list_of_sentences)
    for confusion_set in list_of_confusion_sets:
        key = ','.join(confusion_set)
        sentences_by_confusion_set[key] = list()
        for word in words:
            if word in confusion_set:
                sentences_by_confusion_set[key].extend(sentences_by_word[word])
    return sentences_by_confusion_set


def replace_confusion_set_words_in_sentences(list_of_sentences_by_confusion_sets):
    result = dict()
    nlp = spacy.load("en_core_web_sm")
    for confusion_set, sentences in list_of_sentences_by_confusion_sets.items():
        tmp_sentences = []
        list_of_words = confusion_set.split(',')
        for word in list_of_words:
            for partner in list_of_words:
                if word != partner:
                    for sentence in sentences:
                        doc = nlp(sentence)
                        for token in doc:
                            if token.text == word:
                                tmp_sentence = sentence[:token.idx] + partner + sentence[token.idx+len(token.text):]
                                tmp_sentences.append(tmp_sentence)
        result[confusion_set] = tmp_sentences
    return result





