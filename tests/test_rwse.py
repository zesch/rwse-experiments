import experiments.rwse as rwse
import cassis
import spacy

def test_set_confusion_sets():
    checker = rwse.RWSE_Checker()
    expected = dict({
        "and":["and", "und"],
        "und":["and", "und"],
        "dream":["dream", "team", "mean"],
        "team":["dream", "team", "mean"],
        "mean":["dream", "team", "mean"]
    })

    checker.set_confusion_sets([("and", "und"), ("dream", "team", "mean")])
    assert checker.confusion_sets == expected, "list input failed"

    checker.confusion_sets = None
    checker.set_confusion_sets(expected)
    assert checker.confusion_sets == expected, "dictionary input failed"

    checker.confusion_sets = None
    checker.set_confusion_sets("tests/test-data/confusion_sets.csv")
    assert checker.confusion_sets == expected, "string input failed"

def test_check():
    checker = rwse.RWSE_Checker()
    checker.set_confusion_sets([("there", "their")])
    token = "there"
    masked_sentence = "I want to buy [MASK] cars."
    correct_token = checker.check(token, masked_sentence)
    assert correct_token == "their", "incorrect token"


def test_check_cas():
    checker = rwse.RWSE_Checker()
    checker.set_confusion_sets([("there", "their")])

    path = 'tests/test-data/TypeSystem.xml'

    with open(path, 'rb') as f:
        ts = cassis.load_typesystem(f)
    cas = cassis.Cas(ts)
    cas.sofa_string = "I want to buy there cars."

    T_RWSE = 'de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.RWSE'
    T_SENTENCE = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'
    T_TOKEN = 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token'

    nlp = spacy.load('en_core_web_sm')

    S = ts.get_type(T_SENTENCE)
    T = ts.get_type(T_TOKEN)

    doc = nlp(cas.sofa_string)
    for sent in doc.sents:
        cas_sentence = S(begin=sent.start_char, end=sent.end_char)
        cas.add(cas_sentence)
    for token in doc:
        cas_token = T(begin=token.idx, end=token.idx + len(token.text), id=token.i)
        cas.add(cas_token)

    checker.check_cas(cas, ts)
    result = cas.select(T_RWSE)
    assert len(result) == 1, "no RWSE found"
    assert result[0].suggestion == "their", "RWSE check failed"