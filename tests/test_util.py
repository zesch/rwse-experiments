from experiments import util

def test_collect_sentences_by_confusion_sets():
    list_of_confusion_sets = [('and', 'und')]
    list_of_sentences = ["sentence1 and", "sentence2 und", "sentence3"]
    expected = {'and,und': ['sentence2 und', 'sentence1 and']}
    result = util.collect_sentences_by_confusion_sets(list_of_confusion_sets, list_of_sentences)
    assert sorted(result['and,und']) == sorted(expected['and,und'])

def test_replace_confusion_set_words_in_sentences():
    confusion_set = 'and,und'
    list_of_sentences = ["sentence1 and", "sentence2 und"]
    sentences_by_confusion_sets = {confusion_set: list_of_sentences}
    expected = {'and,und': ['sentence2 and', 'sentence1 und']}
    result = util.replace_confusion_set_words_in_sentences(sentences_by_confusion_sets)
    assert sorted(result['and,und']) == sorted(expected['and,und'])