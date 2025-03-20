from rwse import RWSE_Checker

ranges = list(range(1, 10)) + list(range(10, 101, 10))
rwse = RWSE_Checker()
input_file_name = 'experiments/input/eng_news_2023_balanced-masked-sentences.csv'


def run_falsified_data_old():
    result = dict()
    for line in lines:  # skip header
        confusion_set, expected, sentence = line.strip().split('\t')
        rwse.set_confusion_sets([set(confusion_set.split(','))])
        result.setdefault(confusion_set, {'num_sentences': 0, 'num_matches': 0})
        for target in confusion_set.split(','):
            if target != expected:  # analyze mistakes only
                # TODO clean sentence?
                suggestion, _ = rwse.check(target, sentence)
                result[confusion_set]['num_sentences'] += 1
                if suggestion.lower() == expected.lower():  # no case discrimination in RWSE result, true positive
                    result[confusion_set]['num_matches'] += 1
    return result


def run_original_data_old():
    result = dict()
    for line in lines: # skip header
        confusion_set, target, sentence = line.strip().split('\t')
        rwse.set_confusion_sets([set(confusion_set.split(','))])
        # TODO clean sentence?
        suggestion, _ = rwse.check(target, sentence, magnitude=magnitude)
        result.setdefault(confusion_set, {'num_sentences':0, 'num_matches':0})
        result[confusion_set]['num_sentences'] += 1
        if suggestion.lower() != target.lower(): # no case discrimination in RWSE result
            result[confusion_set]['num_matches'] += 1
    return result

def run_falsified_data():
    result = dict()
    for line in lines:  # skip header
        confusion_set, expected, sentence = line.strip().split('\t')
        rwse.set_confusion_sets([set(confusion_set.split(','))])
        for target in confusion_set.split(','):
            if target != expected:  # analyze mistakes only
                # TODO clean sentence?
                results = rwse.check(target, sentence, return_all=True)
                for magnitude in ranges:
                    suggestion, _ = rwse.evaluate(target, results, magnitude)
                    result.setdefault(magnitude, {'num_sentences': 0, 'num_matches': 0})
                    result[magnitude]['num_sentences'] += 1
                    if suggestion.lower() == expected.lower():  # no case discrimination in RWSE result, true positive
                        result[magnitude]['num_matches'] += 1
    return result


def run_original_data():
    result = dict()
    for line in lines: # skip header
        confusion_set, target, sentence = line.strip().split('\t')
        rwse.set_confusion_sets([set(confusion_set.split(','))])
        # TODO clean sentence?
        results = rwse.check(target, sentence, return_all=True)
        for magnitude in ranges:
            suggestion, _ = rwse.evaluate(target, results, magnitude)
            result.setdefault(magnitude, {'num_sentences':0, 'num_matches':0})
            result[magnitude]['num_sentences'] += 1
            if suggestion.lower() != target.lower(): # no case discrimination in RWSE result
                result[magnitude]['num_matches'] += 1
    return result


if __name__ == "__main__":
    magnitude_search = dict()
    with open(input_file_name, 'r') as f:
        lines = f.readlines()[1:] #TODO

    print('Starting with false positives.')
    false_positive_result = run_original_data()
    print('Starting with true positives.')
    true_positive_result = run_falsified_data()

    with open('experiments/output/report_magnitude_search_balanced.csv', 'w') as output_file:

        print('magnitude', 'TP', 'TN', 'FP', 'FN', 'total', sep=';', file=output_file)
        for magnitude in ranges:

            print(f'Printing magnitude {magnitude}...')

            TP, TN, FP, FN = 0, 0, 0, 0
            total = 0
            num_sentences = false_positive_result[magnitude]['num_sentences']
            num_matches = false_positive_result[magnitude]['num_matches']
            #num_sentences = sum([value['num_sentences'] for value in false_positive_result.values()])
            #num_matches = sum([value['num_matches'] for value in false_positive_result.values()])
            #print('false_positives', num_sentences, num_matches)
            TN = num_sentences - num_matches
            FP = num_matches
            total += num_sentences

            num_sentences = true_positive_result[magnitude]['num_sentences']
            num_matches = true_positive_result[magnitude]['num_matches']
            #num_sentences = sum([value['num_sentences'] for value in true_positive_result.values()])
            #num_matches = sum([value['num_matches'] for value in true_positive_result.values()])
            #print('true_positives', num_sentences, num_matches)
            FN = num_sentences - num_matches
            TP = num_matches
            total += num_sentences

            print(magnitude, TP, TN, FP, FN, total, sep=';', file=output_file)
