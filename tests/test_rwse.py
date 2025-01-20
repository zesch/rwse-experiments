from rwse import rwse

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
