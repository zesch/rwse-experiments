from experiments.util import helper
import spacy

nlp = spacy.load("en_core_web_sm")

def test_collect_sentences_by_confusion_sets():
    list_of_confusion_sets = [('and', 'und')]
    list_of_sentences = ["sentence1 and", "sentence2 und", "sentence3"]
    expected = {'and,und': ['sentence2 und', 'sentence1 and']}
    result = helper.collect_sentences_by_confusion_sets(list_of_confusion_sets, list_of_sentences, nlp)
    assert sorted(result['and,und']) == sorted(expected['and,und'])

def test_collect_sentences_by_confusion_sets_no_duplicates():
    list_of_confusion_sets = [('and', 'und')]
    list_of_sentences = ["sentence1 and", "sentence2 und", "sentence1 and", "sentence3"]
    expected = {'and,und': ['sentence2 und', 'sentence1 and']}
    result = helper.collect_sentences_by_confusion_sets(list_of_confusion_sets, list_of_sentences, nlp)
    assert sorted(result['and,und']) == sorted(expected['and,und'])


def test_replace_confusion_set_words_in_sentences():
    confusion_set = 'and,und'
    list_of_sentences = ["sentence1 and", "sentence2 und"]
    sentences_by_confusion_sets = {confusion_set: list_of_sentences}
    expected = {'and,und': ['sentence2 and', 'sentence1 und']}
    result = helper.replace_confusion_set_words_in_sentences(sentences_by_confusion_sets, nlp)
    result_sentences = [item['sentence'] for item in result['and,und']]
    assert sorted(result_sentences)== sorted(expected['and,und'])

def test_replace_confusion_set_words_in_sentences_case_insensitive():
    confusion_set = 'and,And'
    list_of_sentences = ["sentence1 and", "sentence3 And"]
    sentences_by_confusion_sets = {confusion_set: list_of_sentences}
    result = helper.replace_confusion_set_words_in_sentences(sentences_by_confusion_sets, nlp)
    assert len(result[confusion_set]) == 0, 'no sentences should have been modified'

    confusion_set = 'and,And,und'
    list_of_sentences = ["sentence1 and", "sentence3 And"]
    expected = ["sentence1 und", "sentence3 und"]

    sentences_by_confusion_sets = {confusion_set: list_of_sentences}
    result = helper.replace_confusion_set_words_in_sentences(sentences_by_confusion_sets, nlp)
    assert len(result[confusion_set]) == 2, 'only two sentences should have been modified for <und>'
    result_sentences = [item['sentence'] for item in result[confusion_set]]
    assert sorted(result_sentences) == sorted(expected), 'sentences for <und> were not modified correctly'
