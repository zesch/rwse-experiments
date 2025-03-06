from transformers import AutoTokenizer
from helper import check_token

checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

with open('../input/confusion_sets.csv') as f:
    confusion_sets = f.readlines()

    for confusion_set in confusion_sets:
        words = [item.strip() for item in confusion_set.split(',')]
        words.extend([item.capitalize() for item in words])
        words = sorted([item for item in set(words) if check_token(tokenizer, item) is not None])
        if len(words) > 2 or len(words) == 2 and words[0].lower() != words[1].lower():
            with open('../input/confusion_sets_modified.csv', 'a') as nf:
                nf.write(','.join(sorted(words)) + '\n')
        else:
            print(f'No partners for {confusion_set.strip()}. Result is {words}')

