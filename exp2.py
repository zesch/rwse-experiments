from transformers import pipeline

pipe = pipeline("fill-mask", model="bert-base-cased")
#results = pipe("I want to buy [MASK] car", targets=["their", "there"])
results = pipe("People with lots of [MASK] usually live in big houses.", targets=["honey", "money"])

for result in results:
    print(f'{result["token_str"]}: {result["score"]}')