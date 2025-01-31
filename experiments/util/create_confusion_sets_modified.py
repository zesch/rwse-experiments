from transformers import AutoTokenizer

checkpoint = 'bert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def check_token(target:str) -> bool:
    target_enc = tokenizer.tokenize(target)
    if len(target_enc) > 1 or target_enc[0] == tokenizer.unk_token:
        return None
    return target

with open('../input/confusion_sets.csv') as f:
    confusion_sets = f.readlines()
    for confusion_set in confusion_sets:
        words = [item.strip() for item in confusion_set.split(',')]
        words.extend([item.capitalize() for item in words])
        words = sorted([item for item in set(words) if check_token(item) is not None])
        if len(words) > 2 or len(words) == 2 and words[0].lower() != words[1].lower():
            with open('../input/confusion_sets_modified.csv', 'a') as nf:
                nf.write(','.join(sorted(words)) + '\n')
        else:
            print(f'No partners for {confusion_set.strip()}. Result is {words}')

